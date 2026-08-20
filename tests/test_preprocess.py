import pytest

from src.api.main import clean_text


def test_clean_text_standard_urls():
    assert clean_text("Check http://example.com/test now") == "Check now"
    assert clean_text("Visit https://sub.domain.org/path?query=1&arg=2") == "Visit"
    assert clean_text("Go to www.google.com for info") == "Go to for info"
    assert clean_text("Old marker <url> replaced") == "Old marker replaced"


def test_clean_text_uppercase_urls():
    assert clean_text("Link HTTPS://EXAMPLE.COM/PATH here") == "Link here"
    assert clean_text("Link WWW.MYPAGE.ORG here") == "Link here"


def test_clean_text_reddit_artifacts():
    assert clean_text("Posted on r/depression by u/alex_99") == "Posted on by"
    assert clean_text("Message [deleted] and post [removed]") == "Message and post"
    assert clean_text("Check R/ANXIETY and U/MODERATOR") == "Check and"


def test_clean_text_html_entities():
    assert clean_text("Feeling down &amp; hopeless") == "Feeling down hopeless"
    assert clean_text("Don&#39;t worry &quot;friend&quot;") == 'Don\'t worry "friend"'
    assert clean_text("&lt;b&gt;bold text&lt;/b&gt;") == "b bold text b"
    assert clean_text("First line&nbsp;second line") == "First line second line"


def test_clean_text_whitespace_and_tabs():
    assert clean_text("   Leading and trailing   ") == "Leading and trailing"
    assert clean_text("Line 1\n\n\nLine 2\t\tLine 3") == "Line 1 Line 2 Line 3"
    assert clean_text("Multiple       consecutive        spaces") == "Multiple consecutive spaces"


def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text("     ") == ""
    assert clean_text("\n\t\r") == ""
    assert clean_text(None) == ""


def test_clean_text_special_characters_and_emojis():
    cleaned = clean_text("Anxious! #stress @work %tired &lost *scared")
    assert "Anxious" in cleaned
    assert "stress" in cleaned
    assert "work" in cleaned
    assert "tired" in cleaned
    assert "scared" in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "%" not in cleaned


def test_clean_text_large_payload():
    large_input = "I am feeling overwhelmed. " * 2000
    cleaned = clean_text(large_input)
    assert len(cleaned) > 10000
    assert "overwhelmed" in cleaned
