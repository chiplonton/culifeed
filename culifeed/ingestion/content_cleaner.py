"""
Content Cleaner
===============

HTML content cleaning and text extraction utilities for RSS feeds.

This module provides:
- HTML tag removal and text extraction
- Content sanitization and normalization
- Link extraction and processing
- Text quality validation
- Security-focused HTML cleaning
"""

import re
import html
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment, NavigableString
from bs4.element import CData, ProcessingInstruction, Doctype

from culifeed.config.settings import get_settings
from culifeed.utils.logging import get_logger_for_component
from culifeed.utils.validators import validate_url
from culifeed.utils.exceptions import ContentValidationError


class ContentCleaner:
    """
    HTML content cleaner with security focus and text extraction.

    Features:
    - Removes dangerous HTML elements and attributes
    - Extracts clean text content from HTML
    - Preserves useful formatting while removing clutter
    - Validates and normalizes URLs
    - Handles various encoding issues
    """

    # HTML elements to completely remove (including content)
    DANGEROUS_ELEMENTS = {
        "script",
        "style",
        "iframe",
        "embed",
        "object",
        "applet",
        "form",
        "input",
        "button",
        "select",
        "textarea",
        "meta",
        "link",
        "base",
        "noscript",
        "canvas",
    }

    # HTML elements to remove but preserve content
    FORMATTING_ELEMENTS = {
        "font",
        "center",
        "big",
        "small",
        "tt",
        "strike",
        "s",
        "u",
        "blink",
        "marquee",
        "nobr",
        "wbr",
    }

    # HTML elements that are safe to keep
    SAFE_ELEMENTS = {
        "p",
        "br",
        "div",
        "span",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "strong",
        "b",
        "em",
        "i",
        "code",
        "pre",
        "blockquote",
        "q",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "table",
        "tr",
        "td",
        "th",
        "thead",
        "tbody",
        "tfoot",
        "caption",
        "a",
        "img",
    }

    # Attributes to keep for specific elements
    SAFE_ATTRIBUTES = {
        "a": ["href", "title"],
        "img": ["src", "alt", "title", "width", "height"],
        "blockquote": ["cite"],
        "q": ["cite"],
    }

    # Patterns for cleaning text
    WHITESPACE_PATTERN = re.compile(r"\s+", re.MULTILINE)
    MULTIPLE_NEWLINES_PATTERN = re.compile(r"\n\s*\n\s*\n+", re.MULTILINE)
    LEADING_TRAILING_PATTERN = re.compile(r"^\s+|\s+$", re.MULTILINE)

    # URL validation patterns
    JAVASCRIPT_URL_PATTERN = re.compile(r"^\s*javascript:", re.IGNORECASE)
    DATA_URL_PATTERN = re.compile(r"^\s*data:", re.IGNORECASE)

    def __init__(self):
        """Initialize content cleaner with settings."""
        self.settings = get_settings()
        self.logger = get_logger_for_component("content_cleaner")

        # Configure BeautifulSoup parser
        self.parser = "html.parser"  # Built-in parser, no external deps

    def clean_html_content(
        self, html_content: str, base_url: Optional[str] = None
    ) -> str:
        """
        Clean HTML content and extract readable text.

        Args:
            html_content: Raw HTML content to clean
            base_url: Base URL for resolving relative links

        Returns:
            Cleaned plain text content

        Raises:
            ContentValidationError: If content is invalid or dangerous
        """
        if not html_content or not html_content.strip():
            return ""

        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, self.parser)

            # Remove dangerous elements completely
            self._remove_dangerous_elements(soup)

            # Remove comments, CDATA, processing instructions
            self._remove_non_content_elements(soup)

            # Clean and validate attributes
            self._clean_attributes(soup, base_url)

            # Remove formatting-only elements but keep content
            self._unwrap_formatting_elements(soup)

            # Extract clean text
            cleaned_text = self._extract_text_content(soup)

            # Normalize whitespace and formatting
            cleaned_text = self._normalize_text(cleaned_text)

            # Validate final content
            if not self._validate_cleaned_content(cleaned_text):
                raise ContentValidationError("Content failed validation after cleaning")

            self.logger.debug(
                f"Cleaned HTML: {len(html_content)} -> {len(cleaned_text)} chars"
            )
            return cleaned_text

        except Exception as e:
            self.logger.error(f"Failed to clean HTML content: {e}")
            # Return a basic text extraction as fallback
            return self._extract_text_fallback(html_content)

    def extract_text_only(self, html_content: str) -> str:
        """
        Extract only text content from HTML, removing all markup.

        Args:
            html_content: HTML content to process

        Returns:
            Plain text content with all HTML removed
        """
        if not html_content or not html_content.strip():
            return ""

        try:
            soup = BeautifulSoup(html_content, self.parser)

            # Remove script, style, and other non-content elements
            for element in soup(self.DANGEROUS_ELEMENTS):
                element.decompose()

            # Get all text content
            text = soup.get_text(separator=" ", strip=True)

            # Normalize whitespace
            text = self.WHITESPACE_PATTERN.sub(" ", text)
            text = text.strip()

            return text

        except Exception as e:
            self.logger.warning(f"Failed to extract text, using fallback: {e}")
            return self._extract_text_fallback(html_content)

    def extract_links(
        self, html_content: str, base_url: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Extract all valid links from HTML content.

        Args:
            html_content: HTML content to process
            base_url: Base URL for resolving relative links

        Returns:
            List of dictionaries with 'url', 'text', and 'title' keys
        """
        links = []

        if not html_content or not html_content.strip():
            return links

        try:
            soup = BeautifulSoup(html_content, self.parser)

            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href", "").strip()

                if not href:
                    continue

                # Skip javascript and data URLs
                if self.JAVASCRIPT_URL_PATTERN.match(
                    href
                ) or self.DATA_URL_PATTERN.match(href):
                    continue

                # Resolve relative URLs
                if base_url and not urlparse(href).netloc:
                    href = urljoin(base_url, href)

                # Validate URL
                if not validate_url(href):
                    continue

                # Extract link text and title
                link_text = a_tag.get_text(strip=True) or ""
                link_title = a_tag.get("title", "").strip() or ""

                links.append({"url": href, "text": link_text, "title": link_title})

        except Exception as e:
            self.logger.warning(f"Failed to extract links: {e}")

        return links

    def extract_images(
        self, html_content: str, base_url: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Extract image information from HTML content.

        Args:
            html_content: HTML content to process
            base_url: Base URL for resolving relative URLs

        Returns:
            List of dictionaries with image information
        """
        images = []

        if not html_content or not html_content.strip():
            return images

        try:
            soup = BeautifulSoup(html_content, self.parser)

            for img_tag in soup.find_all("img", src=True):
                src = img_tag.get("src", "").strip()

                if not src:
                    continue

                # Skip data URLs (base64 images)
                if self.DATA_URL_PATTERN.match(src):
                    continue

                # Resolve relative URLs
                if base_url and not urlparse(src).netloc:
                    src = urljoin(base_url, src)

                # Validate URL
                if not validate_url(src):
                    continue

                # Extract image metadata
                alt_text = img_tag.get("alt", "").strip() or ""
                title = img_tag.get("title", "").strip() or ""
                width = img_tag.get("width", "") or ""
                height = img_tag.get("height", "") or ""

                images.append(
                    {
                        "src": src,
                        "alt": alt_text,
                        "title": title,
                        "width": width,
                        "height": height,
                    }
                )

        except Exception as e:
            self.logger.warning(f"Failed to extract images: {e}")

        return images

    def clean_and_extract_metadata(
        self, html_content: str, base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Clean HTML and extract both text and metadata.

        Args:
            html_content: HTML content to process
            base_url: Base URL for resolving relative URLs

        Returns:
            Dictionary with cleaned text, links, images, and other metadata
        """
        result = {
            "clean_text": "",
            "links": [],
            "images": [],
            "word_count": 0,
            "has_media": False,
        }

        if not html_content or not html_content.strip():
            return result

        try:
            # Clean text content
            result["clean_text"] = self.clean_html_content(html_content, base_url)
            result["word_count"] = len(result["clean_text"].split())

            # Extract links and images
            result["links"] = self.extract_links(html_content, base_url)
            result["images"] = self.extract_images(html_content, base_url)

            # Check for media content
            result["has_media"] = len(result["images"]) > 0 or len(result["links"]) > 0

        except Exception as e:
            self.logger.error(f"Failed to clean and extract metadata: {e}")
            # Fallback to basic text extraction
            result["clean_text"] = self.extract_text_only(html_content)
            result["word_count"] = len(result["clean_text"].split())

        return result

    def _remove_dangerous_elements(self, soup: BeautifulSoup) -> None:
        """Remove dangerous HTML elements completely."""
        for element_name in self.DANGEROUS_ELEMENTS:
            for element in soup.find_all(element_name):
                element.decompose()

    def _remove_non_content_elements(self, soup: BeautifulSoup) -> None:
        """Remove comments, CDATA, and processing instructions."""
        for element in soup(
            text=lambda text: isinstance(
                text, (Comment, CData, ProcessingInstruction, Doctype)
            )
        ):
            element.extract()

    def _clean_attributes(
        self, soup: BeautifulSoup, base_url: Optional[str] = None
    ) -> None:
        """Clean and validate element attributes."""
        for element in soup.find_all():
            element_name = element.name.lower()
            safe_attrs = self.SAFE_ATTRIBUTES.get(element_name, [])

            # Remove all attributes except safe ones
            attrs_to_remove = []
            for attr_name in element.attrs:
                if attr_name.lower() not in safe_attrs:
                    attrs_to_remove.append(attr_name)

            for attr_name in attrs_to_remove:
                del element[attr_name]

            # Validate and clean remaining attributes
            if element_name == "a" and "href" in element.attrs:
                href = element.get("href", "").strip()

                # Remove javascript and data URLs
                if self.JAVASCRIPT_URL_PATTERN.match(
                    href
                ) or self.DATA_URL_PATTERN.match(href):
                    del element["href"]
                elif base_url and not urlparse(href).netloc:
                    # Convert relative to absolute URL
                    element["href"] = urljoin(base_url, href)

            elif element_name == "img" and "src" in element.attrs:
                src = element.get("src", "").strip()

                # Skip data URLs
                if self.DATA_URL_PATTERN.match(src):
                    element.decompose()
                elif base_url and not urlparse(src).netloc:
                    # Convert relative to absolute URL
                    element["src"] = urljoin(base_url, src)

    def _unwrap_formatting_elements(self, soup: BeautifulSoup) -> None:
        """Remove formatting elements but keep their content."""
        for element_name in self.FORMATTING_ELEMENTS:
            for element in soup.find_all(element_name):
                element.unwrap()

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from soup."""
        # Use get_text with separator to preserve some structure
        text = soup.get_text(separator="\n", strip=True)
        return text

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and formatting in text."""
        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Normalize whitespace within lines
        text = self.WHITESPACE_PATTERN.sub(" ", text)

        # Limit consecutive newlines
        text = self.MULTIPLE_NEWLINES_PATTERN.sub("\n\n", text)

        # Clean leading/trailing whitespace from each line
        lines = text.split("\n")
        cleaned_lines = [line.strip() for line in lines]

        # Remove empty lines at start and end, limit consecutive empty lines
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()

        # Join lines back together
        text = "\n".join(cleaned_lines)

        return text.strip()

    def _validate_cleaned_content(self, content: str) -> bool:
        """Validate that cleaned content meets quality criteria."""
        if not content or len(content.strip()) < 10:
            return False

        # Check for reasonable text-to-whitespace ratio
        text_chars = len(re.sub(r"\s", "", content))
        total_chars = len(content)

        if total_chars > 0 and (text_chars / total_chars) < 0.1:
            return False

        # Check content length limits
        max_length = self.settings.processing.max_content_length * 3
        if len(content) > max_length:
            self.logger.warning(f"Content too long: {len(content)} > {max_length}")
            return False

        return True

    def _extract_text_fallback(self, html_content: str) -> str:
        """Fallback text extraction using regex when BeautifulSoup fails."""
        try:
            # Remove script and style tags with content
            content = re.sub(
                r"<(script|style)[^>]*>.*?</\1>",
                "",
                html_content,
                flags=re.IGNORECASE | re.DOTALL,
            )

            # Remove all HTML tags
            content = re.sub(r"<[^>]+>", "", content)

            # Decode HTML entities
            content = html.unescape(content)

            # Normalize whitespace
            content = self.WHITESPACE_PATTERN.sub(" ", content)
            content = content.strip()

            return content

        except Exception as e:
            self.logger.error(f"Fallback text extraction failed: {e}")
            return ""


# Convenience functions for common operations
def clean_html_text(html_content: str, base_url: Optional[str] = None) -> str:
    """Quick function to clean HTML and extract text."""
    cleaner = ContentCleaner()
    return cleaner.clean_html_content(html_content, base_url)


def extract_plain_text(html_content: str) -> str:
    """Quick function to extract plain text from HTML."""
    cleaner = ContentCleaner()
    return cleaner.extract_text_only(html_content)


def extract_content_metadata(
    html_content: str, base_url: Optional[str] = None
) -> Dict[str, Any]:
    """Quick function to extract text and metadata from HTML."""
    cleaner = ContentCleaner()
    return cleaner.clean_and_extract_metadata(html_content, base_url)
