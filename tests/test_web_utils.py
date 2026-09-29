"""Tests for web scraping (ingestion/web_crawling.py) and chunk caching (ingestion/web_utils.py)."""

import pytest
from bs4 import BeautifulSoup

import config
from ingestion import web_crawling
from ingestion.web_crawling import create_session, crawl_website, scrape_webpage
from ingestion.web_utils import save_or_load_web_chunks

TEST_URLS = {
    "Data management plans": config.WEB_URLS["Data management plans"],
}

# TU Delft pages put their content in "t3ce frame-type-text" blocks; the last two blocks are page boilerplate
TUDELFT_PAGE = """
<html><body>
  <nav><p>Navigation menu that must be ignored</p></nav>
  <div class="t3ce frame-type-text"><p>A data management plan describes how data is handled.</p></div>
  <div class="t3ce frame-type-text"><ul><li>Store data on TU Delft storage.</li></ul></div>
  <div class="t3ce frame-type-text"><p>Footer block one, must be ignored.</p></div>
  <div class="t3ce frame-type-text"><p>Footer block two, must be ignored.</p></div>
</body></html>
"""


def simple_split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def crawl_single_page(url, visited):
    """Crawl only the start page, to keep network tests fast."""
    return crawl_website(url, max_pages=1, delay=0, visited=visited)


def test_scrape_webpage_extracts_content_blocks():
    text = scrape_webpage(BeautifulSoup(TUDELFT_PAGE, "html.parser"))
    assert "data management plan describes" in text
    assert "Store data on TU Delft storage." in text
    assert "Navigation menu" not in text
    assert "Footer block" not in text


def test_scrape_webpage_without_content_blocks_returns_empty():
    assert scrape_webpage("<html><body><p>No TU Delft content here</p></body></html>") == ""


class FakeResponse:
    status_code = 200
    headers = {"Content-Type": "text/html"}

    def __init__(self, html):
        self.content = html.encode()


class FakeSite:
    """A fake website: every page has content and links to 5 child pages on the same domain."""

    def __init__(self):
        self.fetched = []

    def get(self, url, timeout=None):
        self.fetched.append(url)
        links = "".join(f'<a href="{url.rstrip("/")}/{i}">child</a>' for i in range(5))
        return FakeResponse(f'<div class="t3ce frame-type-text"><p>Content of {url}</p></div>{links}')


def test_every_start_url_gets_its_own_page_budget(monkeypatch):
    site = FakeSite()
    monkeypatch.setattr(web_crawling, "create_session", lambda: site)

    visited = set()
    first = crawl_website("https://example.org/a", max_pages=3, delay=0, visited=visited)
    second = crawl_website("https://example.org/b", max_pages=3, delay=0, visited=visited)

    assert len(first) == 3
    assert len(second) == 3, "the second start URL must not be skipped because of pages crawled from the first"
    assert second[0][0] == "https://example.org/b"


def test_already_visited_pages_are_not_fetched_again(monkeypatch):
    site = FakeSite()
    monkeypatch.setattr(web_crawling, "create_session", lambda: site)

    visited = set()
    crawl_website("https://example.org/a", max_pages=2, delay=0, visited=visited)
    crawl_website("https://example.org/a", max_pages=2, delay=0, visited=visited)

    assert len(site.fetched) == len(set(site.fetched))


def test_save_or_load_web_chunks_removes_duplicate_chunks(tmp_path):
    pages = {"https://example.org/a": "Same text.", "https://example.org/b": "Same text.", "https://example.org/c": "Other text."}

    def fake_crawler(url, visited):
        return [(url, pages[url])]

    chunks = save_or_load_web_chunks(
        str(tmp_path / "web_chunks.pkl"), {u: u for u in pages}, lambda text: [text], web_crawling_func=fake_crawler
    )
    assert chunks == ["Same text.", "Other text."]


@pytest.mark.network
@pytest.mark.parametrize("url", TEST_URLS.values())
def test_scrape_live_webpage_returns_text(url):
    response = create_session().get(url, timeout=10)
    response.raise_for_status()
    text = scrape_webpage(BeautifulSoup(response.content, "html.parser"))
    assert text.strip(), f"no text scraped from {url}"

    chunks = simple_split_text(text)
    assert chunks
    assert all(len(c) <= 500 for c in chunks)


@pytest.mark.network
def test_save_or_load_web_chunks_creates_and_reloads_pickle(tmp_path):
    pickle_path = tmp_path / "web_chunks.pkl"

    # First call scrapes the pages and writes the pickle
    chunks_first = save_or_load_web_chunks(
        str(pickle_path), TEST_URLS, simple_split_text, web_crawling_func=crawl_single_page
    )
    assert pickle_path.exists()
    assert chunks_first

    # Second call must load straight from the pickle, without crawling
    def fail_if_called(*args, **kwargs):
        raise AssertionError("crawler called although the pickle exists")

    chunks_second = save_or_load_web_chunks(
        str(pickle_path), TEST_URLS, simple_split_text, web_crawling_func=fail_if_called
    )
    assert chunks_first == chunks_second
