import argparse
import asyncio
import csv
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from multiprocessing import cpu_count

import aiohttp
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s @ %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
logger = logging.getLogger(name="LentaParser")


class LentaParser:

    # lxml is much faster but error prone
    default_parser = "html.parser"

    def __init__(self, *, max_workers: int, outfile_name: str, from_date: str):
        self._endpoint = "https://lenta.ru/news"

        self._sess = None
        self._connector = None

        self._executor = ProcessPoolExecutor(max_workers=max_workers)

        self._outfile_name = outfile_name
        self._outfile = None
        self._csv_writer = None
        self.timeouts = aiohttp.ClientTimeout(total=60, connect=60)

        self._n_downloaded = 0
        self._from_date = datetime.strptime(from_date, "%d.%m.%Y")

    @property
    def dates_countdown(self):
        date_start, date_end = self._from_date, datetime.today()

        while date_start <= date_end:
            yield date_start.strftime("%Y/%m/%d")
            date_start += timedelta(days=1)

    @property
    def writer(self):
        if self._csv_writer is None:
            self._outfile = open(self._outfile_name, "w", encoding="utf-8", newline="")
            self._csv_writer = csv.DictWriter(
                self._outfile, fieldnames=["url", "title", "text", "topic", "tags",
                "subheading", "author", "num_links", "num_images"]
            )
            self._csv_writer.writeheader()

        return self._csv_writer

    @property
    def session(self):
        if self._sess is None or self._sess.closed:

            self._connector = aiohttp.TCPConnector(
                use_dns_cache=True, ttl_dns_cache=60 * 60, limit=1024
            )
            self._sess = aiohttp.ClientSession(
                connector=self._connector, timeout=self.timeouts
            )

        return self._sess

    async def fetch(self, url: str):
        response = await self.session.get(url, allow_redirects=False)
        response.raise_for_status()
        return await response.text(encoding="utf-8")

    @staticmethod
    def parse_article_html(html: str):
        doc_tree = BeautifulSoup(html, LentaParser.default_parser)

        # 1️⃣ Найти текст статьи
        body = doc_tree.find("div", class_="topic-body__content")
        if not body:
            body = doc_tree.find("div", class_="article__text")  # запасной вариант

        if not body:
            raise RuntimeError(f"Article body is not found")

        text = " ".join(p.get_text(separator=" ") for p in body.find_all("p"))

        # 2️⃣ Найти тему
        topic = None
        topic_tag = doc_tree.find("a", class_="topic-header__rubric")
        if topic_tag:
            topic = topic_tag.get_text(strip=True)

        # 3️⃣ Найти заголовок
        title_tag = doc_tree.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else None

        # 4️⃣ Найти теги (если есть)
        tags = None
        tags_block = doc_tree.find("div", class_="tags")
        if tags_block:
            tags = ", ".join(tag.get_text(strip=True) for tag in tags_block.find_all("a"))

        if not tags:
            meta_keywords = doc_tree.find("meta", attrs={"name": "keywords"})
            if meta_keywords and meta_keywords.get("content"):
                tags = meta_keywords["content"]

        # 5️⃣ Подзаголовок (если есть)
        subheading = None
        subtitle_tag = doc_tree.find("h2")
        if subtitle_tag:
            subheading = subtitle_tag.get_text(strip=True)

        # 6️⃣ Автор (если указан)
        author = None
        author_tag = doc_tree.find("div", class_="topic-author")
        if author_tag:
            author = author_tag.get_text(strip=True)

        # 7️⃣ Количество ссылок в тексте
        num_links = len(body.find_all("a")) if body else 0

        # 8️⃣ Количество изображений в статье
        num_images = len(body.find_all("img")) if body else 0

        return {
            "title": title,
            "text": text,
            "topic": topic,
            "tags": tags,
            "subheading": subheading,
            "author": author,
            "num_links": num_links,
            "num_images": num_images
        }


    @staticmethod
    def _extract_urls_from_html(html: str, current_date: str):
        doc_tree = BeautifulSoup(html, LentaParser.default_parser)
        news_links = set()

        for a_tag in doc_tree.find_all("a", href=True):
            href = a_tag["href"]
            if (
                href.startswith(f"/news/{current_date}/") and
                href.count('/') > 3 and
                not "/page/" in href  # <-- фильтр пагинации
            ):
                news_links.add(f"https://lenta.ru{href}")

        return tuple(news_links)



    async def _fetch_all_news_on_page(self, html: str, current_date: str):
        # Get news URLs from raw html
        loop = asyncio.get_running_loop()
        news_urls = await loop.run_in_executor(
        self._executor, self._extract_urls_from_html, html, current_date
    )

        # Fetching news
        tasks = tuple(asyncio.create_task(self.fetch(url)) for url in news_urls)

        fetched_raw_news = dict()

        for i, task in enumerate(tasks):
            try:
                fetch_res = await task
            except aiohttp.ClientResponseError as exc:
                logger.error(f"Cannot fetch {exc.request_info.url}: {exc}")
            except asyncio.TimeoutError:
                logger.exception("Cannot fetch. Timout")
            else:
                fetched_raw_news[news_urls[i]] = fetch_res

        for url, html in fetched_raw_news.items():
            fetched_raw_news[url] = loop.run_in_executor(
                self._executor, self.parse_article_html, html
            )

        parsed_news = []

        for url, task in fetched_raw_news.items():
            try:
                parse_res = await task
            except Exception:
                logger.exception(f"Cannot parse {url}")
            else:
                parse_res["url"] = url
                parsed_news.append(parse_res)

        if parsed_news:
            self.writer.writerows(parsed_news)
            self._n_downloaded += len(parsed_news)

        return len(parsed_news)

    async def shutdown(self):
        if self._sess is not None:
            await self._sess.close()

        await asyncio.sleep(0.5)

        if self._outfile is not None:
            self._outfile.close()

        self._executor.shutdown(wait=True)

        logger.info(f"{self._n_downloaded} news saved at {self._outfile_name}")

    async def _producer(self):
        for date in self.dates_countdown:
            page_num = 1

            while True:
                suffix = f"/page/{page_num}/" if page_num > 1 else ""
                news_page_url = f"{self._endpoint}/{date}{suffix}"

                try:
                    html = await asyncio.create_task(self.fetch(news_page_url))
                except aiohttp.ClientResponseError as e:
                    if e.status == 404:
                        break  # Страница закончилась
                    logger.exception(f"Cannot fetch {news_page_url}")
                    break
                except aiohttp.ClientConnectionError:
                    logger.exception(f"Cannot fetch {news_page_url}")
                    break
                else:
                    n_proccessed_news = await self._fetch_all_news_on_page(html, date)

                    if n_proccessed_news == 0:
                        logger.info(f"No articles found on {news_page_url}.")
                    else:
                        logger.info(
                            f"{news_page_url} processed ({n_proccessed_news} news). "
                            f"{self._n_downloaded} news saved totally."
                        )

                page_num += 1  # Переходим к следующей странице дня

    async def run(self):
        try:
            await self._producer()
        finally:
            await self.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Downloads news from Lenta.Ru")

    parser.add_argument(
        "--outfile", default="lenta-ru-news.csv", help="name of result file"
    )

    parser.add_argument(
        "--cpu-workers", default=cpu_count(), type=int, help="number of workers"
    )

    parser.add_argument(
        "--from-date",
        default="01.05.2025",
        type=str,
        help="download news from this date. Example: 30.08.1999",
    )

    args = parser.parse_args()

    parser = LentaParser(
        max_workers=args.cpu_workers,
        outfile_name=args.outfile,
        from_date=args.from_date,
    )

    try:
        asyncio.run(parser.run())
    except KeyboardInterrupt:
        asyncio.run(parser.shutdown())
        logger.info("KeyboardInterrupt, exiting...")


if __name__ == "__main__":
    main()
