import unittest
import sys
import os

# 1. Add your project folder to the system path so Python finds 'backend.py'
PROJECT_PATH = r"C:\Users\fleX Gaming\Downloads\project"
sys.path.insert(0, PROJECT_PATH)

from backend import Backend

class TestBackend(unittest.TestCase):
    def setUp(self):
        """Initializes the Backend before every test."""
        self.backend = Backend()

    def test_files_exist(self):
        """Test if critical CSV files are actually found."""
        # We check if the CPU data loaded correctly
        self.assertIn("CPU", self.backend.parts_data, "CPU data should be loaded in backend")
        self.assertFalse(self.backend.parts_data["CPU"].empty, "CPU CSV should not be empty")

    def test_budget_logic(self):
        """Test if the recommendation stays within budget."""
        target_budget = 2000
        results = self.backend.recommend_builds_rule_based(target_budget)
        
        if results:
            first_build_cost = results[0]['cost']
            print(f"\nTested Budget: ${target_budget}, Build Cost: ${first_build_cost}")
            self.assertLessEqual(first_build_cost, target_budget, "Build cost implies the budget was ignored!")
        else:
            print("\nNo builds found (this is valid if data is missing, but check CSVs).")

if __name__ == '__main__':
    unittest.main()