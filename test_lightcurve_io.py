import unittest

from lightcurve_io import ProductDownloadError, download_search_result, looks_transient


class FakeCollection:
    def __init__(self, curves):
        self.curves = curves


class FakeProduct:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = 0

    def download(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.value


class FakeSearch:
    def __init__(self, batch=None, batch_error=None, products=()):
        self.batch = batch
        self.batch_error = batch_error
        self.products = list(products)
        self.download_all_calls = 0

    def __len__(self):
        return len(self.products)

    def __getitem__(self, index):
        return self.products[index]

    def download_all(self, **kwargs):
        self.download_all_calls += 1
        if self.batch_error:
            raise self.batch_error
        return self.batch


class FakeLightkurve:
    LightCurveCollection = FakeCollection


class LightcurveIoTests(unittest.TestCase):
    def test_request_failed_is_retryable(self):
        self.assertTrue(looks_transient("Request failed"))
        self.assertTrue(looks_transient("HTTP 503 Service Unavailable"))
        self.assertFalse(looks_transient("invalid target identifier"))

    def test_batch_success_avoids_individual_downloads(self):
        expected = FakeCollection(["batch"])
        product = FakeProduct("individual")
        result = download_search_result(
            FakeSearch(batch=expected, products=[product]),
            FakeLightkurve, "cache", "hard",
        )
        self.assertIs(result, expected)
        self.assertEqual(product.calls, 0)

    def test_batch_failure_keeps_individually_available_products(self):
        product_ok = FakeProduct("sector-105")
        product_bad = FakeProduct(error=OSError("Request failed"))
        result = download_search_result(
            FakeSearch(
                batch_error=OSError("Request failed"),
                products=[product_ok, product_bad],
            ),
            FakeLightkurve, "cache", "hard",
        )
        self.assertEqual(result.curves, ["sector-105"])
        self.assertEqual(product_ok.calls, 1)
        self.assertEqual(product_bad.calls, 1)

    def test_all_failures_include_batch_and_individual_details(self):
        search = FakeSearch(
            batch_error=OSError("batch Request failed"),
            products=[FakeProduct(error=OSError("product Request failed"))],
        )
        with self.assertRaisesRegex(ProductDownloadError, "product 0"):
            download_search_result(search, FakeLightkurve, "cache", "hard")


if __name__ == "__main__":
    unittest.main()
