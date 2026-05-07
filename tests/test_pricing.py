import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from environment import HighDimBlackScholes

class TestDeepBSDEPricing(unittest.TestCase):
    
    def test_analytical_benchmark_baseline(self):
        market = HighDimBlackScholes(dim=100, total_time=1.0, num_time_intervals=20)
        
        calculated_price = market.analytical_pricing()
        
        expected_price = 2.1192
        
        self.assertAlmostEqual(
            calculated_price, 
            expected_price, 
            places=4, 
            msg=f"Pricing Error: Expected {expected_price}, but got {calculated_price:.4f}"
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
