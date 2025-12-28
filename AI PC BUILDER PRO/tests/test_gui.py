import unittest
import tkinter as tk
import sys

# 1. Add your project folder to the system path
PROJECT_PATH = r"C:\Users\fleX Gaming\Downloads\project"
sys.path.insert(0, PROJECT_PATH)

from app import PCBuilderApp

class TestGUI(unittest.TestCase):
    def test_interface_launch(self):
        """Test if the App window initializes without crashing."""
        root = tk.Tk()
        try:
            app = PCBuilderApp(root)
            self.assertEqual(root.title(), "AI PC Builder Pro", "Title does not match!")
            print("\nGUI launched successfully.")
        except Exception as e:
            self.fail(f"GUI crashed on startup: {e}")
        finally:
            root.destroy() # Close the window immediately

if __name__ == '__main__':
    unittest.main()