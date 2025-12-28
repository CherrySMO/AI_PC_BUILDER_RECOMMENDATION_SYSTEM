import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import requests
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import reportlab
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd # pandas is crucial for data handling and checks

# Import the Backend class directly from backend.py
# Assuming backend.py is in the same directory or accessible via Python path
from backend import Backend

# === Configuration (retained in frontend as they relate to UI display or are global constants) ===
# IMPORTANT: This BASE_PATH should point to where your CSV data files and ML models are stored.
# If they are in a subfolder named 'data' relative to where this frontend.py file is,
# then 'data' is correct. Otherwise, adjust this path.
BASE_PATH = os.path.join(os.path.dirname(__file__), "data") # Ensure this path is correct
PART_FILES = {
    "CPU": "CPU.csv",
    "GPU": "GPU.csv",
    "Motherboard": "Motherboard.csv",
    "RAM": "Memory.csv",
    "Storage": "Storage.csv",
    "PSU": "PSU.csv",
    "Case": "Case.csv",
    "Cooler": "Cooler.csv"
}

SCENARIOS = ["Gaming", "Workstation", "Content Creation", "Home Office", "General Use"]
IMAGE_SIZE = (80, 80)

# Define which numerical columns to use for feature graphs for each part type.
# These are selected to be meaningful for a Radar Chart or an improved Bar Chart.
# Only truly numerical features are included for graphing.
FEATURE_COLUMNS_FOR_GRAPHS = {
    "CPU": ["speed", "coreCount", "threadCount", "power", "price"],
    "GPU": ["VRAM", "power", "price"],
    "RAM": ["size", "price"],
    "Storage": ["space", "price"],
    "PSU": ["power", "price"],
    "Motherboard": ["price", "sataPorts", "pcieSlots", "usbPorts"],
    "Case": ["price"],
    "Cooler": ["price"]
}

# Specific features to display for each part type in "Key Specifications" tables/details.
SPEC_MAP = {
    "CPU": ["socket", "speed", "coreCount", "threadCount", "power"],
    "Motherboard": ["socket", "size"],
    "RAM": ["type", "size"],
    "Storage": ["type", "space"],
    "PSU": ["power", "size"],
    "Case": ["size"],
    "Cooler": ["type"]
}

# Modern Color Scheme for the UI
COLORS = {
    'primary': '#1a1a2e',      # Dark navy
    'secondary': '#16213e',    # Darker blue
    'accent': '#0f3460',       # Medium blue
    'highlight': '#533483',    # Purple (Used as one component color)
    'success': '#00d4aa',      # Teal (Used as another component color)
    'warning': '#f39c12',      # Orange
    'danger': '#e74c3c',       # Red
    'light': '#f8f9fa',        # Light gray
    'medium': '#6c757d',       # Medium gray
    'dark': '#212529',         # Dark gray
    'card': '#ffffff',         # White (for card backgrounds)

    # Specific colors for chart elements
    'background': '#F8FAFC', # Lightest background for chart
    'grid': '#E2E8F0',       # Light grid lines
    'text': '#1E293B',       # Dark text for labels
}

class ModernButton(tk.Button):
    """
    A custom Tkinter button with modern styling and hover effects.
    """
    def __init__(self, parent, text="", command=None, bg_color=COLORS['highlight'],
                 hover_color=COLORS['success'], text_color='white', **kwargs):
        super().__init__(parent, text=text, command=command, **kwargs)

        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color

        self.config(
            bg=self.bg_color,
            fg=self.text_color,
            font=('Segoe UI', 10, 'bold'),
            relief='flat', # Flat appearance
            bd=0,          # No border
            padx=20,       # Horizontal padding
            pady=10,       # Vertical padding
            cursor='hand2' # Change cursor on hover
        )

        # Bind hover events
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        """Changes button background on mouse enter."""
        self.config(bg=self.hover_color)

    def on_leave(self, e):
        """Restores button background on mouse leave."""
        self.config(bg=self.bg_color)

class ModernCard(tk.Frame):
    """
    A custom Tkinter frame designed to look like a modern card, with a subtle shadow.
    """
    def __init__(self, parent, bg_color=COLORS['card'], **kwargs):
        super().__init__(parent, bg=bg_color, relief='flat', bd=0, **kwargs)

        # Add a subtle "shadow" by placing a slightly darker frame behind and slightly offset
        shadow = tk.Frame(parent, bg=COLORS['medium'], height=2)
        shadow.place(in_=self, x=2, y=2, relwidth=1, relheight=1)
        self.lift() # Ensure the card frame is on top of its shadow

class PCBuilderApp:
    """
    The main frontend application class for the AI PC Builder Pro.
    Manages the Tkinter GUI and interacts with the Backend for data and logic.
    """
    def __init__(self, root):
        """
        Initializes the main application, setting up the GUI and backend connection.
        """
        self.root = root
        self.root.title("AI PC Builder Pro")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS['primary'])
        self.root.state('zoomed')  # Start maximized on Windows

        # Initialize the Backend: This is the critical connection to your backend logic.
        # Pass BASE_PATH and PART_FILES so the Backend can load its data.
        self.backend = Backend(BASE_PATH, PART_FILES)

        # Retrieve essential data and models from the backend
        self.parts_data = self.backend.get_parts_data() # Dictionary of DataFrames for each part type
        self.models = self.backend.ml_models             # Dictionary of loaded ML models
        self.normalization_ranges = self.backend.get_normalization_ranges() # Min/max ranges for features

        # State variables for the GUI
        self.recommendations = []        # Stores rule-based recommendations
        self.smart_recommendations = []  # Stores smart/ML-based recommendations
        self.image_cache = {}            # Cache for loaded images
        self.current_custom_build = {}   # Holds the components of the currently customized build

        # Variables to hold references to opened Toplevel windows
        self.compare_window = None
        self.graph_window = None

        # Setup GUI styles and layout
        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        """Configures the Tkinter ttk.Style for a modern look and feel."""
        style = ttk.Style()
        style.theme_use('clam') # Use 'clam' theme as a base

        # Configure various widget styles
        style.configure('Modern.TFrame', background=COLORS['card'])
        style.configure('Sidebar.TFrame', background=COLORS['secondary'])
        style.configure('Header.TLabel',
                        font=('Segoe UI', 24, 'bold'),
                        background=COLORS['secondary'],
                        foreground=COLORS['light'])
        style.configure('SubHeader.TLabel',
                        font=('Segoe UI', 16, 'bold'),
                        background=COLORS['card'],
                        foreground=COLORS['dark'])
        style.configure('Card.TLabel',
                        font=('Segoe UI', 11),
                        background=COLORS['card'],
                        foreground=COLORS['dark'])
        style.configure('Modern.Treeview',
                        font=('Segoe UI', 10),
                        rowheight=30,
                        fieldbackground=COLORS['light'])
        style.map('Modern.Treeview',
                  background=[('selected', COLORS['accent'])],
                  foreground=[('selected', 'white')])
        style.configure('Modern.Treeview.Heading',
                        font=('Segoe UI', 11, 'bold'),
                        background=COLORS['accent'],
                        foreground='white')

        # Combobox style
        style.configure('Modern.TCombobox',
                        fieldbackground=COLORS['light'],
                        background=COLORS['accent'],
                        foreground=COLORS['dark'],
                        arrowcolor=COLORS['dark'],
                        selectbackground=COLORS['highlight'],
                        selectforeground='white',
                        font=('Segoe UI', 10))
        style.map('Modern.TCombobox',
                  fieldbackground=[('readonly', COLORS['light'])],
                  selectbackground=[('readonly', COLORS['highlight'])])
        
        # Notebook (tabbed interface) style
        style.configure('TNotebook', background=COLORS['light'], borderwidth=0)
        style.configure('TNotebook.Tab', background=COLORS['secondary'], foreground=COLORS['light'],
                        font=('Segoe UI', 10, 'bold'), padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', COLORS['accent'])],
                  foreground=[('selected', 'white')])

    def setup_ui(self):
        """Sets up the main layout of the application with header, sidebar, and tabbed content."""
        # Main container frame
        main_container = tk.Frame(self.root, bg=COLORS['primary'])
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Application Header
        self.create_header(main_container)

        # Content area (holds sidebar and main notebook)
        content_frame = tk.Frame(main_container, bg=COLORS['primary'])
        content_frame.pack(fill='both', expand=True, pady=(20, 0))

        # Left Sidebar
        self.create_sidebar(content_frame)

        # Main content area: A Notebook (tabbed interface)
        self.main_content_notebook = ttk.Notebook(content_frame, style='TNotebook')
        self.main_content_notebook.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        # Create and add each tab
        self.recommendations_tab = tk.Frame(self.main_content_notebook, bg=COLORS['light'])
        self.main_content_notebook.add(self.recommendations_tab, text="Build Recommendations")
        self.create_recommendations_tab_content() # Populate this tab

        self.smart_recommendation_tab = tk.Frame(self.main_content_notebook, bg=COLORS['light'])
        self.main_content_notebook.add(self.smart_recommendation_tab, text="Smart Recommendations")
        self.create_smart_recommendation_tab_content(self.smart_recommendation_tab) # Populate this tab

        self.custom_build_tab = tk.Frame(self.main_content_notebook, bg=COLORS['light'])
        self.main_content_notebook.add(self.custom_build_tab, text="🛠️ Custom Build")
        self.create_custom_build_tab_content(self.custom_build_tab) # Populate this tab

        self.compare_tab = tk.Frame(self.main_content_notebook, bg=COLORS['light'])
        self.main_content_notebook.add(self.compare_tab, text="🔬 Component Comparison")
        self.create_compare_tab_content(self.compare_tab) # Populate this tab

        self.graphs_tab = tk.Frame(self.main_content_notebook, bg=COLORS['light'])
        self.main_content_notebook.add(self.graphs_tab, text="📊 Graphs")
        self.create_graphs_tab_content(self.graphs_tab) # Populate this tab

        # Display an initial welcome screen in the recommendations tab
        self.create_welcome_screen(self.recommendations_frame_content)

    def create_header(self, parent):
        """Creates the top header bar of the application."""
        header_frame = ModernCard(parent, bg_color=COLORS['secondary'])
        header_frame.pack(fill='x', pady=(0, 20))

        title_frame = tk.Frame(header_frame, bg=COLORS['secondary'])
        title_frame.pack(fill='x', padx=30, pady=20)

        title_label = tk.Label(title_frame,
                               text="🖥️ AI PC Builder Pro",
                               font=('Segoe UI', 28, 'bold'),
                               bg=COLORS['secondary'],
                               fg=COLORS['light'])
        title_label.pack(side='left')

        subtitle = tk.Label(title_frame,
                            text="Build Your Perfect PC with AI-Powered Recommendations",
                            font=('Segoe UI', 12),
                            bg=COLORS['secondary'],
                            fg=COLORS['medium'])
        subtitle.pack(side='left', padx=(20, 0), pady=(10, 0))

        # System status indicator
        status_frame = tk.Frame(title_frame, bg=COLORS['secondary'])
        status_frame.pack(side='right')

        status_dot = tk.Label(status_frame, text="●",
                              font=('Segoe UI', 16),
                              fg=COLORS['success'], # Green dot for "ready"
                              bg=COLORS['secondary'])
        status_dot.pack(side='right')

        status_text = tk.Label(status_frame, text="System Ready",
                               font=('Segoe UI', 10),
                               fg=COLORS['light'],
                               bg=COLORS['secondary'])
        status_text.pack(side='right', padx=(0, 5))

    def create_sidebar(self, parent):
        """Creates the left sidebar containing budget input and quick stats."""
        sidebar_container = ModernCard(parent, bg_color=COLORS['secondary'])
        sidebar_container.pack(side='left', fill='y', padx=(0, 20))

        sidebar_frame = tk.Frame(sidebar_container, bg=COLORS['secondary'], width=350)
        sidebar_frame.pack(fill='both', expand=True, padx=20, pady=20)
        sidebar_frame.pack_propagate(False) # Prevent frame from shrinking below specified width/height

        # Budget input section
        budget_section = tk.Frame(sidebar_frame, bg=COLORS['secondary'])
        budget_section.pack(fill='x', pady=(0, 30))

        budget_title = tk.Label(budget_section,
                               text="💰 Set Your Budget",
                               font=('Segoe UI', 16, 'bold'),
                               bg=COLORS['secondary'],
                               fg=COLORS['light'])
        budget_title.pack(anchor='w', pady=(0, 10))

        budget_input_frame = tk.Frame(budget_section, bg=COLORS['secondary'])
        budget_input_frame.pack(fill='x', pady=(0, 10))

        dollar_label = tk.Label(budget_input_frame, text="$",
                               font=('Segoe UI', 14, 'bold'),
                               bg=COLORS['secondary'],
                               fg=COLORS['success'])
        dollar_label.pack(side='left', padx=(0, 5))

        self.budget_var = tk.StringVar()
        self.budget_entry = tk.Entry(budget_input_frame,
                                     textvariable=self.budget_var,
                                     font=('Segoe UI', 14),
                                     bg=COLORS['light'],
                                     fg=COLORS['dark'],
                                     relief='flat',
                                     bd=0,
                                     insertbackground=COLORS['dark'],
                                     width=15)
        self.budget_entry.pack(side='left', fill='x', expand=True, ipady=8, padx=(0, 10))
        self.budget_entry.bind("<Return>", lambda e: self.recommend_builds()) # Trigger on Enter key

        # Quick budget suggestion buttons
        suggestions_frame = tk.Frame(budget_section, bg=COLORS['secondary'])
        suggestions_frame.pack(fill='x', pady=(10, 0))

        suggestion_label = tk.Label(suggestions_frame,
                                    text="Quick Suggestions:",
                                    font=('Segoe UI', 10),
                                    bg=COLORS['secondary'],
                                    fg=COLORS['medium'])
        suggestion_label.pack(anchor='w')

        suggestions = [("Budget", "800"), ("Mid-Range", "1500"), ("High-End", "3000")]
        for name, amount in suggestions:
            btn = tk.Button(suggestions_frame,
                            text=f"{name} (${amount})",
                            command=lambda a=amount: self.set_budget(a),
                            bg=COLORS['accent'],
                            fg='white',
                            font=('Segoe UI', 9),
                            relief='flat',
                            bd=0,
                            padx=10,
                            pady=5,
                            cursor='hand2')
            btn.pack(side='left', padx=(0, 5), pady=5)

        # Generate Builds button
        self.generate_btn = ModernButton(sidebar_frame,
                                         text="🚀 Generate Builds",
                                         command=self.recommend_builds,
                                         bg_color=COLORS['success'],
                                         hover_color=COLORS['highlight'])
        self.generate_btn.pack(fill='x', pady=(20, 30))
        
        # Compare Components button (switches to the comparison tab)
        self.compare_btn = ModernButton(sidebar_frame,
                                        text="🔬 Compare Components",
                                        command=self.open_comparison_module,
                                        bg_color=COLORS['accent'],
                                        hover_color=COLORS['highlight'])
        self.compare_btn.pack(fill='x', pady=(10, 20))

        # Quick Stats section (displaying counts of available parts)
        stats_section = ModernCard(sidebar_frame, bg_color=COLORS['card'])
        stats_section.pack(fill='x', pady=(0, 20))

        stats_title = tk.Label(stats_section,
                               text="📊 Quick Stats",
                               font=('Segoe UI', 14, 'bold'),
                               bg=COLORS['card'],
                               fg=COLORS['dark'])
        stats_title.pack(anchor='w', padx=15, pady=(15, 10))

        # Scrollable area for stats
        stats_canvas = tk.Canvas(stats_section, bg=COLORS['card'], highlightthickness=0)
        stats_scrollbar = ttk.Scrollbar(stats_section, orient="vertical", command=stats_canvas.yview)
        
        stats_inner = tk.Frame(stats_canvas, bg=COLORS['card'])
        
        stats_canvas.create_window((0, 0), window=stats_inner, anchor="nw")
        stats_canvas.configure(yscrollcommand=stats_scrollbar.set)

        # Update scrollregion when inner frame size changes
        stats_inner.bind(
            "<Configure>",
            lambda e: stats_canvas.configure(scrollregion=stats_canvas.bbox("all"))
        )

        stats_canvas.pack(side="left", fill="both", expand=True, padx=15, pady=(0, 15))
        stats_scrollbar.pack(side="right", fill="y", pady=(0, 15))

        # Populate stats with data from backend
        for part_type, df in self.parts_data.items():
            if not df.empty:
                stat_frame = tk.Frame(stats_inner, bg=COLORS['card'])
                stat_frame.pack(fill='x', pady=2)

                part_label = tk.Label(stat_frame,
                                      text=f"{part_type}:",
                                      font=('Segoe UI', 10),
                                      bg=COLORS['card'],
                                      fg=COLORS['medium'])
                part_label.pack(side='left')

                count_label = tk.Label(stat_frame,
                                       text=f"{len(df)} options",
                                       font=('Segoe UI', 10, 'bold'),
                                       bg=COLORS['card'],
                                       fg=COLORS['dark'])
                count_label.pack(side='right')

        # Enable mouse wheel scrolling for the stats canvas
        stats_canvas.bind_all("<MouseWheel>", lambda e: stats_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    def set_budget(self, amount):
        """Sets the budget entry field to a predefined amount."""
        self.budget_var.set(amount)
        self.budget_entry.focus_set() # Set focus to the entry

    def create_recommendations_tab_content(self):
        """Sets up the content area for the 'Build Recommendations' tab."""
        self.recommendations_frame_content = tk.Frame(self.recommendations_tab, bg=COLORS['light'])
        self.recommendations_frame_content.pack(fill='both', expand=True)

    def create_welcome_screen(self, parent_frame):
        """Displays a welcome message and features list on an empty tab."""
        for widget in parent_frame.winfo_children():
            widget.destroy() # Clear any existing content

        welcome_frame = tk.Frame(parent_frame, bg=COLORS['light'])
        welcome_frame.pack(fill='both', expand=True, padx=40, pady=40)

        # Center content using place
        center_frame = tk.Frame(welcome_frame, bg=COLORS['light'])
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        welcome_icon = tk.Label(center_frame, text="🎯", font=('Segoe UI', 80), bg=COLORS['light'])
        welcome_icon.pack(pady=(0, 20))

        welcome_title = tk.Label(center_frame, text="Ready to Build Your Dream PC?", font=('Segoe UI', 24, 'bold'), bg=COLORS['light'], fg=COLORS['dark'])
        welcome_title.pack(pady=(0, 10))

        welcome_subtitle = tk.Label(center_frame, text="Enter your budget and let our AI find the perfect components for you", font=('Segoe UI', 14), bg=COLORS['light'], fg=COLORS['medium'])
        welcome_subtitle.pack(pady=(0, 30))

        # List of key features
        features = [
            "🔍 Smart component matching",
            "💡 Multiple build options",
            "📊 Detailed performance analysis",
            "📄 PDF export capability",
            "🤖 Rule-based Smart Recommendations"
        ]

        for feature in features:
            feature_label = tk.Label(center_frame, text=feature, font=('Segoe UI', 12), bg=COLORS['light'], fg=COLORS['dark'])
            feature_label.pack(pady=5)

    def recommend_builds(self):
        """
        Triggers the rule-based build recommendation process.
        Validates budget, shows loading, calls backend, and displays results.
        """
        for widget in self.recommendations_frame_content.winfo_children():
            widget.destroy() # Clear previous recommendations

        try:
            budget = float(self.budget_var.get())
            if budget <= 0:
                messagebox.showerror("Invalid Input", "Budget must be a positive number.")
                self.create_welcome_screen(self.recommendations_frame_content)
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric budget.")
            self.create_welcome_screen(self.recommendations_frame_content)
            return

        self.show_loading_screen(self.recommendations_frame_content) # Show loading indicator

        # Call the backend for rule-based recommendations
        self.recommendations = self.backend.recommend_builds_rule_based(budget)

        if not self.recommendations:
            messagebox.showinfo("No Builds Found", "No complete builds possible with this budget. Try a higher budget.")
            self.create_welcome_screen(self.recommendations_frame_content) # Show welcome if no builds
        else:
            self.display_modern_recommendations(self.recommendations_frame_content) # Display results

    def show_loading_screen(self, parent_frame):
        """Displays a loading animation/message while processing."""
        for widget in parent_frame.winfo_children():
            widget.destroy()

        loading_frame = tk.Frame(parent_frame, bg=COLORS['light'])
        loading_frame.pack(fill='both', expand=True)

        center_frame = tk.Frame(loading_frame, bg=COLORS['light'])
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        loading_label = tk.Label(center_frame, text="🔄 Analyzing Components...", font=('Segoe UI', 20, 'bold'), bg=COLORS['light'], fg=COLORS['dark'])
        loading_label.pack()

        self.root.update_idletasks() # Update GUI immediately to show loading

    def display_modern_recommendations(self, parent_frame):
        """
        Displays the generated PC build recommendations in a scrollable, card-based layout.
        """
        for widget in parent_frame.winfo_children():
            widget.destroy()

        # Create a scrollable area for the recommendation cards
        canvas = tk.Canvas(parent_frame, bg=COLORS['light'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['light'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")) # Update scroll region on frame resize
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Header for the recommendations display
        header_frame = tk.Frame(scrollable_frame, bg=COLORS['light'])
        header_frame.pack(fill='x', padx=30, pady=(20, 30))

        title = tk.Label(header_frame, text="🎯 Your Recommended Builds", font=('Segoe UI', 24, 'bold'), bg=COLORS['light'], fg=COLORS['dark'])
        title.pack(side='left')

        # Create a card for each recommendation
        for idx, rec in enumerate(self.recommendations):
            self.create_build_card(scrollable_frame, rec, idx)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling on the canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def create_build_card(self, parent, recommendation, index):
        """
        Creates a visual card for a single PC build recommendation,
        including details, cost, and action buttons.
        """
        card_container = tk.Frame(parent, bg=COLORS['light'])
        card_container.pack(fill='x', padx=30, pady=(0, 25))

        card = ModernCard(card_container, bg_color=COLORS['card'])
        card.pack(fill='x')

        card_content = tk.Frame(card, bg=COLORS['card'])
        card_content.pack(fill='both', expand=True, padx=25, pady=20)

        # Card Header: Build Type and Cost
        header_frame = tk.Frame(card_content, bg=COLORS['card'])
        header_frame.pack(fill='x', pady=(0, 15))

        # Handle different title formats for rule-based vs. smart builds
        build_title_text = recommendation.get('type', f'Build {index + 1}') if isinstance(index, int) else f"Smart Build: {index}"
        build_title = tk.Label(header_frame, text=build_title_text, font=('Segoe UI', 18, 'bold'), bg=COLORS['card'], fg=COLORS['dark'])
        build_title.pack(side='left')

        cost_frame = tk.Frame(header_frame, bg=COLORS['card'])
        cost_frame.pack(side='right')

        total_cost = tk.Label(cost_frame, text=f"${recommendation['cost']:.2f}", font=('Segoe UI', 20, 'bold'), bg=COLORS['card'], fg=COLORS['success'])
        total_cost.pack(side='right')

        remaining = tk.Label(cost_frame, text=f"Remaining: ${recommendation['remaining']:.2f}", font=('Segoe UI', 11), bg=COLORS['card'], fg=COLORS['medium'])
        remaining.pack(side='right', padx=(0, 15))

        # Components grid: Displays each part in the build
        components_frame = tk.Frame(card_content, bg=COLORS['card'])
        components_frame.pack(fill='x', pady=(0, 20))

        row = 0
        col = 0
        for part_type, part_data in recommendation['parts'].items():
            if part_type == '_ml_score': # Skip internal ML score if present
                continue
            self.create_component_item(components_frame, part_type, part_data, row, col)
            col += 1
            if col >= 2: # Max 2 columns per row
                col = 0
                row += 1

        # Action buttons for the build card
        button_frame = tk.Frame(card_content, bg=COLORS['card'])
        button_frame.pack(fill='x', pady=(15, 0))

        customize_btn = ModernButton(button_frame, text="🔧 Customize",
                                     command=lambda: self.open_modern_customizer(recommendation, index),
                                     bg_color=COLORS['highlight'], hover_color=COLORS['success'])
        customize_btn.pack(side='left', padx=(0, 10))

        export_btn = ModernButton(button_frame, text="📄 Export PDF",
                                  command=lambda r=recommendation, idx=index: self.export_build_to_pdf(r, idx),
                                  bg_color=COLORS['warning'], hover_color=COLORS['danger'])
        export_btn.pack(side='left')

        view_details_btn = ModernButton(button_frame, text="📊 View Details",
                                        command=lambda r=recommendation, idx=index: self.show_build_details(r, idx),
                                        bg_color=COLORS['accent'], hover_color=COLORS['highlight'])
        view_details_btn.pack(side='right')

    def create_component_item(self, parent, part_type, part_data, row, col):
        """
        Creates a small display item for a single component within a build card.
        """
        item_frame = tk.Frame(parent, bg=COLORS['card'], relief='flat', bd=1, highlightbackground=COLORS['medium'], highlightthickness=1)
        item_frame.grid(row=row, column=col, padx=10, pady=8, sticky='ew')
        parent.columnconfigure(col, weight=1) # Allow columns to expand

        content_frame = tk.Frame(item_frame, bg=COLORS['card'])
        content_frame.pack(fill='both', expand=True, padx=10, pady=8)

        type_label = tk.Label(content_frame, text=part_type, font=('Segoe UI', 10, 'bold'), bg=COLORS['card'], fg=COLORS['accent'])
        type_label.pack(anchor='w')

        name_label = tk.Label(content_frame, text=part_data['name'][:40] + "..." if len(part_data['name']) > 40 else part_data['name'], font=('Segoe UI', 9), bg=COLORS['card'], fg=COLORS['dark'], wraplength=200)
        name_label.pack(anchor='w')

        price_label = tk.Label(content_frame, text=f"${part_data['price']:.2f}", font=('Segoe UI', 10, 'bold'), bg=COLORS['card'], fg=COLORS['success'])
        price_label.pack(anchor='w')

    def show_build_details(self, recommendation, index):
        """
        Opens a new Toplevel window to display detailed specifications for a selected build.
        """
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Build {index} Details" if isinstance(index, str) else f"Build {index + 1} Details")
        details_window.geometry("900x700")
        details_window.configure(bg=COLORS['light'])
        details_window.transient(self.root) # Make it modal to the main window
        details_window.grab_set() # Prevent interaction with main window until closed

        # Header for the details window
        header_frame = tk.Frame(details_window, bg=COLORS['secondary'])
        header_frame.pack(fill='x')

        title = tk.Label(header_frame, text=f"📊 {recommendation.get('type', f'Build {index + 1}')} Details" if isinstance(index, int) else f"📊 Smart Build: {index} Details", font=('Segoe UI', 20, 'bold'), bg=COLORS['secondary'], fg=COLORS['light'])
        title.pack(pady=20)

        content_frame = tk.Frame(details_window, bg=COLORS['light'])
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Treeview to display component specifications in a tabular format
        tree_frame = tk.Frame(content_frame, bg=COLORS['light'])
        tree_frame.pack(fill='both', expand=True)

        columns = ("Component", "Specifications", "Price")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", style="Modern.Treeview")

        tree.heading("Component", text="Component")
        tree.heading("Specifications", text="Key Specifications")
        tree.heading("Price", text="Price")

        tree.column("Component", width=200)
        tree.column("Specifications", width=400)
        tree.column("Price", width=100)

        for part_type, part_data in recommendation['parts'].items():
            if part_type == '_ml_score':
                continue

            specs_to_display = []
            if part_type in SPEC_MAP: # Use SPEC_MAP for defined features
                for spec_key in SPEC_MAP[part_type]:
                    value = part_data.get(spec_key)
                    if value is not None and str(value) != 'nan':
                        display_key = ' '.join(word.capitalize() for word in spec_key.split('_'))
                        specs_to_display.append(f"{display_key}: {value}")
            elif part_type == "GPU": # Special handling for GPU VRAM/power if not in SPEC_MAP
                if part_data.get('VRAM') is not None and str(part_data.get('VRAM')) != 'nan':
                     specs_to_display.append(f"VRAM: {part_data['VRAM']}")
                if part_data.get('power') is not None and str(part_data.get('power')) != 'nan':
                     specs_to_display.append(f"Power: {part_data['power']}W")
            else:
                specs_to_display.append("No specific features defined for display.")

            spec_text = " | ".join(specs_to_display) if specs_to_display else "N/A"

            tree.insert("", "end", values=(
                f"{part_type}: {part_data['name']}",
                spec_text,
                f"${part_data['price']:.2f}"
            ))

        tree.pack(fill='both', expand=True)

        # Scrollbars for the treeview
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")

        # Close button
        close_btn = ModernButton(details_window, text="Close", command=details_window.destroy, bg_color=COLORS['accent'], hover_color=COLORS['highlight'])
        close_btn.pack(pady=10)

    def open_modern_customizer(self, recommendation, build_index):
        """
        Opens a new Toplevel window for customizing a selected PC build.
        Allows users to change individual components using dropdowns.
        """
        # Close any existing customizer window
        if hasattr(self, 'customize_window') and self.customize_window.winfo_exists():
            self.customize_window.destroy()

        # Determine window title based on build type
        window_title = f"Customize Build {build_index + 1}" if isinstance(build_index, int) else f"Customize Smart Build: {build_index}"
        
        self.customize_window = tk.Toplevel(self.root)
        self.customize_window.title(window_title)
        self.customize_window.geometry("1000x800")
        self.customize_window.configure(bg=COLORS['light'])
        self.customize_window.transient(self.root) # Make it modal to the main window
        self.customize_window.grab_set()

        self.current_build_index = build_index # Store original index for applying changes
        # Create a deep copy of the parts to allow modification without altering the original recommendation
        self.current_custom_build = {part_type: part_data.copy() for part_type, part_data in recommendation['parts'].items() if part_type != '_ml_score'}
        self.custom_build_vars = {} # Dictionary to hold Tkinter StringVars for comboboxes

        # Header for the customizer window
        header_frame = tk.Frame(self.customize_window, bg=COLORS['secondary'])
        header_frame.pack(fill='x')
        tk.Label(header_frame, text=window_title, font=('Segoe UI', 20, 'bold'), bg=COLORS['secondary'], fg=COLORS['light']).pack(pady=15)

        # Main content area for customization (split into current build display and options)
        main_cust_frame = tk.Frame(self.customize_window, bg=COLORS['light'])
        main_cust_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Left pane: Display current build configuration
        current_build_frame = ModernCard(main_cust_frame, bg_color=COLORS['card'])
        current_build_frame.pack(side='left', fill='both', expand=True, padx=(0, 15))
        current_build_frame.grid_propagate(False) # Prevent frame from shrinking

        tk.Label(current_build_frame, text="Current Build Configuration", font=('Segoe UI', 14, 'bold'), bg=COLORS['card'], fg=COLORS['dark']).pack(pady=(15, 10))

        # Scrollable canvas for current build details
        self.current_build_canvas = tk.Canvas(current_build_frame, bg=COLORS['card'], highlightthickness=0)
        self.current_build_scrollbar = ttk.Scrollbar(current_build_frame, orient="vertical", command=self.current_build_canvas.yview)
        self.current_build_inner_frame = tk.Frame(self.current_build_canvas, bg=COLORS['card'])

        self.current_build_inner_frame.bind(
            "<Configure>",
            lambda e: self.current_build_canvas.configure(
                scrollregion=self.current_build_canvas.bbox("all")
            )
        )
        self.current_build_canvas.create_window((0, 0), window=self.current_build_inner_frame, anchor="nw")
        self.current_build_canvas.configure(yscrollcommand=self.current_build_scrollbar.set)

        self.current_build_canvas.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        self.current_build_scrollbar.pack(side='right', fill='y')

        # Right pane: Customization options (comboboxes for each part type)
        options_frame = ModernCard(main_cust_frame, bg_color=COLORS['card'])
        options_frame.pack(side='right', fill='both', expand=True, padx=(15, 0))

        tk.Label(options_frame, text="Choose Alternative Parts", font=('Segoe UI', 14, 'bold'), bg=COLORS['card'], fg=COLORS['dark']).pack(pady=(15, 10))

        for part_type in PART_FILES.keys():
            if part_type in self.parts_data:
                part_options_frame = tk.Frame(options_frame, bg=COLORS['card'])
                part_options_frame.pack(fill='x', padx=15, pady=5)

                tk.Label(part_options_frame, text=f"{part_type}:", font=('Segoe UI', 10, 'bold'), bg=COLORS['card'], fg=COLORS['accent']).pack(side='left', padx=(0, 10))

                # Get all part names for the dropdown
                part_names = self.parts_data[part_type]['name'].tolist()
                part_var = tk.StringVar()
                self.custom_build_vars[part_type] = part_var

                # Set initial dropdown value to the current part in the build
                if part_type in self.current_custom_build and self.current_custom_build[part_type]['name'] in part_names:
                    part_var.set(self.current_custom_build[part_type]['name'])
                elif part_names:
                    part_var.set(part_names[0]) # Default to first item if current not found

                combobox = ttk.Combobox(part_options_frame, textvariable=part_var, values=part_names,
                                        state='readonly', font=('Segoe UI', 10), style='Modern.TCombobox')
                combobox.pack(side='right', fill='x', expand=True)
                # Bind event to update build when selection changes
                combobox.bind("<<ComboboxSelected>>",
                              lambda event, pt=part_type: self.update_part_selection(pt))

        # Total Cost Display for the custom build
        self.custom_total_cost_label = tk.Label(main_cust_frame,
                                                  text=f"Total Cost: ${self.calculate_custom_build_cost():.2f}",
                                                  font=('Segoe UI', 16, 'bold'), bg=COLORS['light'], fg=COLORS['success'])
        self.custom_total_cost_label.pack(pady=(15, 20))

        # Action Buttons for the customizer
        button_frame = tk.Frame(self.customize_window, bg=COLORS['light'])
        button_frame.pack(pady=10)

        ModernButton(button_frame, text="✅ Apply Changes", command=self.apply_customization,
                     bg_color=COLORS['success'], hover_color=COLORS['highlight']).pack(side='left', padx=5)
        ModernButton(button_frame, text="↩️ Reset", command=lambda: self.reset_customization(recommendation),
                     bg_color=COLORS['accent'], hover_color=COLORS['medium']).pack(side='left', padx=5)
        ModernButton(button_frame, text="📊 Analyze & Compare", command=self.generate_graphs, # This will open a new graph window
                     bg_color=COLORS['highlight'], hover_color=COLORS['warning']).pack(side='left', padx=5)
        ModernButton(button_frame, text="✖️ Close", command=self.customize_window.destroy,
                     bg_color=COLORS['danger'], hover_color=COLORS['dark']).pack(side='right', padx=5)

        self.update_custom_build_display() # Initial display of the custom build

    def update_custom_build_display(self):
        """
        Refreshes the display of the current custom build in the customizer window,
        showing selected parts, images, and updating the total cost.
        """
        for widget in self.current_build_inner_frame.winfo_children():
            widget.destroy() # Clear existing part display

        # Display each part in the current custom build
        for part_type, part_data in self.current_custom_build.items():
            part_frame = tk.Frame(self.current_build_inner_frame, bg=COLORS['card'], bd=1, relief='solid')
            part_frame.pack(fill='x', padx=5, pady=5, ipadx=5, ipady=5)

            # Image display
            image_url = part_data.get('image')
            # Use a generic placeholder image URL if image_url is invalid or missing
            if not image_url or not image_url.startswith(('http://', 'https://')):
                image_url = f"https://placehold.co/{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}/6c757d/f8f9fa?text={part_type}"

            img = self.load_image(image_url, size=(60, 60))
            if img:
                img_label = tk.Label(part_frame, image=img, bg=COLORS['card'])
                img_label.image = img
                img_label.pack(side='left', padx=(0, 10))
            else:
                tk.Label(part_frame, text="No Image", bg=COLORS['card'], fg=COLORS['medium'], font=('Segoe UI', 8)).pack(side='left', padx=(0, 10))


            # Text details for the part
            text_frame = tk.Frame(part_frame, bg=COLORS['card'])
            text_frame.pack(side='left', fill='x', expand=True)

            tk.Label(text_frame, text=f"{part_type}:", font=('Segoe UI', 9, 'bold'),
                     bg=COLORS['card'], fg=COLORS['accent']).pack(anchor='w')
            tk.Label(text_frame, text=f"{part_data['name']}", font=('Segoe UI', 10),
                     bg=COLORS['card'], fg=COLORS['dark'], wraplength=250, justify='left').pack(anchor='w')
            tk.Label(text_frame, text=f"${part_data['price']:.2f}", font=('Segoe UI', 10, 'bold'),
                     bg=COLORS['card'], fg=COLORS['success']).pack(anchor='w')

        # Update the total cost label
        self.custom_total_cost_label.config(text=f"Total Cost: ${self.calculate_custom_build_cost():.2f}")

    def update_part_selection(self, part_type):
        """
        Callback function when a part is selected from a combobox in the customizer.
        Updates the current custom build and refreshes its display.
        """
        selected_part_name = self.custom_build_vars[part_type].get()
        selected_df = self.parts_data[part_type]
        # Find the full data for the selected part
        new_part_data = selected_df[selected_df['name'] == selected_part_name].iloc[0].to_dict()
        self.current_custom_build[part_type] = new_part_data # Update the part in the custom build
        self.update_custom_build_display() # Refresh the display

    def calculate_custom_build_cost(self):
        """Calculates the total cost of the current custom build."""
        return sum(part['price'] for part in self.current_custom_build.values())

    def apply_customization(self):
        """
        Applies the changes made in the customizer to the original recommendation
        (if it was a rule-based build) or simply acknowledges for smart builds.
        """
        new_cost = self.calculate_custom_build_cost()
        original_budget = float(self.budget_var.get())

        if isinstance(self.current_build_index, int) and 0 <= self.current_build_index < len(self.recommendations):
            # Update the original rule-based recommendation list
            self.recommendations[self.current_build_index]['parts'] = self.current_custom_build
            self.recommendations[self.current_build_index]['cost'] = new_cost
            self.recommendations[self.current_build_index]['remaining'] = original_budget - new_cost
            messagebox.showinfo("Success", "Build customized and applied successfully!")
            self.customize_window.destroy()
            self.display_modern_recommendations(self.recommendations_frame_content) # Refresh main display
        elif isinstance(self.current_build_index, str):
            # For smart builds, we don't modify the 'original' in the same way,
            # so just acknowledge the customization. A real app might save this
            # custom smart build as a new entity.
            messagebox.showinfo("Success", "Smart Build customized. Please regenerate the smart recommendation to reflect changes if desired.")
            self.customize_window.destroy()
        else:
            messagebox.showerror("Error", "Could not apply customization. Build index invalid.")

    def reset_customization(self, original_recommendation):
        """Resets the custom build back to its original state from the recommendation."""
        # Revert the current custom build to the original recommendation's parts
        self.current_custom_build = {part_type: part_data.copy() for part_type, part_data in original_recommendation['parts'].items() if part_type != '_ml_score'}
        # Update comboboxes to reflect original selections
        for part_type, part_var in self.custom_build_vars.items():
            if part_type in self.current_custom_build:
                part_var.set(self.current_custom_build[part_type]['name'])
        self.update_custom_build_display() # Refresh the display
        messagebox.showinfo("Reset", "Customization reset to original build.")

    def generate_graphs(self):
        """
        Opens a new Toplevel window to display comparative graphs (bar and radar)
        for the currently customized build and some alternatives.
        """
        # Close any existing graph window
        if self.graph_window and self.graph_window.winfo_exists():
            self.graph_window.destroy()
            self.graph_window = None # Clear reference

        self.graph_window = tk.Toplevel(self.root)
        self.graph_window.title("Performance & Price Comparison")
        self.graph_window.geometry("1200x800") # Increased size for pop-up
        self.graph_window.configure(bg=COLORS['light'])
        self.graph_window.transient(self.root) # Make it modal to the main window
        self.graph_window.grab_set()

        # Handle window close protocol to ensure matplotlib figures are closed
        self.graph_window.protocol("WM_DELETE_WINDOW", self._on_graph_window_close)

        notebook = ttk.Notebook(self.graph_window, style='TNotebook')
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Iterate through each part type in the current custom build
        for part_type, current_part_data in self.current_custom_build.items():
            if part_type in FEATURE_COLUMNS_FOR_GRAPHS: # Only generate graphs for parts with defined features
                features = FEATURE_COLUMNS_FOR_GRAPHS[part_type]
                df_all_parts = self.parts_data.get(part_type)

                if df_all_parts is not None and not df_all_parts.empty:
                    # Filter for valid numerical features that exist in the data
                    valid_features = [f for f in features if f in df_all_parts.columns and pd.api.types.is_numeric_dtype(df_all_parts[f]) and df_all_parts[f].notna().any()]

                    if not valid_features:
                        continue # Skip if no valid numeric features for this part type

                    # Prepare data for plotting: current part + top 3 alternatives
                    plot_data = pd.DataFrame(columns=['name'] + valid_features)

                    current_series = pd.Series(current_part_data)
                    current_series['name'] = f"Selected: {current_series['name']}"
                    plot_data = pd.concat([plot_data, pd.DataFrame([current_series[plot_data.columns]])], ignore_index=True)

                    alternatives = df_all_parts[df_all_parts['name'] != current_part_data['name']].sort_values(by='price', ascending=False).head(3)
                    for _, row in alternatives.iterrows():
                        alt_series = row.copy()
                        alt_series['name'] = f"Alt: {alt_series['name']}"
                        plot_data = pd.concat([plot_data, pd.DataFrame([alt_series[plot_data.columns]])], ignore_index=True)

                    if plot_data.empty:
                        continue

                    tab = tk.Frame(notebook, bg=COLORS['light'])
                    notebook.add(tab, text=f"{part_type} Comparison")

                    fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS['light'])
                    fig.patch.set_facecolor(COLORS['light'])
                    ax.set_facecolor(COLORS['card']) # Plot area background

                    colors = [COLORS['success']] + [COLORS['accent']] * (len(plot_data) - 1)

                    if len(valid_features) > 1:
                        if 'price' in valid_features and len(valid_features) > 1:
                            # Scatter plot price vs. a key performance metric if available
                            performance_metric = [f for f in valid_features if f != 'price' and f != 'power']
                            if performance_metric:
                                x_feature = performance_metric[0]
                                ax.scatter(plot_data[x_feature], plot_data['price'], color=colors, s=150, alpha=0.7)
                                for i, row in plot_data.iterrows():
                                    ax.text(row[x_feature], row['price'], row['name'].replace("Selected: ", "").replace("Alt: ", "")[:15] + "...", fontsize=8, ha='center', va='bottom', color=COLORS['dark'])
                                ax.set_xlabel(x_feature.capitalize(), color=COLORS['dark'])
                                ax.set_ylabel("Price ($)", color=COLORS['dark'])
                                ax.set_title(f"{part_type}: Price vs. {x_feature.capitalize()}", color=COLORS['dark'])
                            else:
                                # Fallback to bar charts for all features if no suitable scatter
                                plot_data.set_index('name').plot(kind='bar', ax=ax, color=colors, legend=False)
                                ax.set_ylabel("Value", color=COLORS['dark'])
                                ax.set_title(f"{part_type} Features Comparison", color=COLORS['dark'])
                                ax.tick_params(axis='x', rotation=45, labelsize=9)
                                ax.tick_params(axis='y', labelsize=9)
                                ax.legend().set_visible(False)
                                for container in ax.containers:
                                    ax.bar_label(container, fmt='%.1f', fontsize=8, color=COLORS['dark'])

                        else:
                            # Just plot all valid features as bar charts
                            plot_data.set_index('name').plot(kind='bar', ax=ax, color=colors, legend=False)
                            ax.set_ylabel("Value", color=COLORS['dark'])
                            ax.set_title(f"{part_type} Features Comparison", color=COLORS['dark'])
                            ax.tick_params(axis='x', rotation=45, labelsize=9)
                            ax.tick_params(axis='y', labelsize=9)
                            ax.legend().set_visible(False)
                            for container in ax.containers:
                                ax.bar_label(container, fmt='%.1f', fontsize=8, color=COLORS['dark'])

                    else: # Only one valid feature (e.g., just price)
                        plot_data.set_index('name').plot(kind='bar', ax=ax, color=colors, legend=False)
                        ax.set_ylabel(valid_features[0].capitalize(), color=COLORS['dark'])
                        ax.set_title(f"{part_type}: {valid_features[0].capitalize()} Comparison", color=COLORS['dark'])
                        ax.tick_params(axis='x', rotation=45, labelsize=9)
                        ax.tick_params(axis='y', labelsize=9)
                        for container in ax.containers:
                            ax.bar_label(container, fmt='%.1f', fontsize=8, color=COLORS['dark'])

                    # General plot styling
                    ax.tick_params(axis='x', colors=COLORS['dark'])
                    ax.tick_params(axis='y', colors=COLORS['dark'])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_color(COLORS['dark'])
                    ax.spines['left'].set_color(COLORS['dark'])

                    # Embed Matplotlib figure into Tkinter
                    canvas = FigureCanvasTkAgg(fig, master=tab)
                    canvas_widget = canvas.get_tk_widget()
                    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

                    toolbar = NavigationToolbar2Tk(canvas, tab) # Matplotlib toolbar for pan/zoom
                    toolbar.update()
                    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                else:
                    tk.Label(tab, text=f"No sufficient data for {part_type} graphs.",
                             bg=COLORS['light'], fg=COLORS['dark'], font=('Segoe UI', 12)).pack(pady=20)
            else:
                # Placeholder if no graphing features are defined for a part type
                tab = tk.Frame(notebook, bg=COLORS['light'])
                notebook.add(tab, text=f"{part_type} (No Graphs)")
                tk.Label(tab, text=f"No specific features defined for {part_type} for graphing.",
                         bg=COLORS['light'], fg=COLORS['dark'], font=('Segoe UI', 12)).pack(pady=20)

        # Close button for the graph window
        close_btn = ModernButton(self.graph_window, text="Close Graphs", command=self._on_graph_window_close,
                                 bg_color=COLORS['danger'], hover_color=COLORS['dark'])
        close_btn.pack(pady=10)
        
        # IMPORTANT: Do not close figures here. Let _on_graph_window_close handle it when the window is closed.
        # plt.close('all') 

    def _on_graph_window_close(self):
        """Handles the closing of the graph window, ensuring Matplotlib figures are closed."""
        if self.graph_window:
            self.graph_window.destroy()
            self.graph_window = None
            plt.close('all') # Close all Matplotlib figures associated with this window


    def _hex_to_rgb(self, hex_color):
        """Converts a hex color string to an RGB tuple (0-1 scale) for ReportLab."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def _wrap_text(self, text, max_chars_per_line):
        """Helper function to wrap text for PDF output, breaking by words."""
        lines = []
        if not text:
            return [""]
        words = text.split(' ')
        current_line = []
        current_len = 0
        for word in words:
            if current_len + len(word) + (1 if current_line else 0) <= max_chars_per_line:
                current_line.append(word)
                current_len += len(word) + (1 if current_line else 0)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_len = len(word)
        lines.append(' '.join(current_line)) # Add the last line
        return lines

    def export_build_to_pdf(self, recommendation, index):
        """
        Exports the details of a given PC build recommendation to a PDF file.
        Includes build summary and component specifications.
        """
        # Determine initial filename based on build type
        initial_file_name_part = f"PC_Build_{recommendation.get('type', f'Build{index+1}').replace(' ', '_').replace('💰', '').replace('⚖️', '').replace('🚀', '')}" if isinstance(index, int) else f"Smart_Build_{str(index).replace(' ', '_').replace('/', '_')}"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title=f"Save Build {index} as PDF",
            initialfile=f"{initial_file_name_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf" # Add timestamp
        )

        if not file_path:
            return # User cancelled

        c = canvas.Canvas(file_path, pagesize=letter)
        width, height = letter # Get page dimensions

        # Header section in PDF
        c.setFillColorRGB(*self._hex_to_rgb(COLORS['secondary']))
        c.rect(0, height - 70, width, 70, fill=True)
        c.setFillColorRGB(*self._hex_to_rgb(COLORS['light']))
        c.setFont('Helvetica-Bold', 24)
        c.drawString(inch, height - 45, "AI PC Builder Pro - Build Recommendation")

        # Build Type and Date
        c.setFillColorRGB(*self._hex_to_rgb(COLORS['dark']))
        c.setFont('Helvetica-Bold', 16)
        build_type_text = recommendation.get('type', f'Build {index + 1}') if isinstance(index, int) else f"Smart Build: {index}"
        c.drawString(inch, height - 100, f"Build Type: {build_type_text}")
        c.setFont('Helvetica', 10)
        c.drawString(inch, height - 120, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        y_position = height - 160

        # Build Summary
        c.setFont('Helvetica-Bold', 14)
        c.drawString(inch, y_position, "Build Summary:")
        y_position -= 20
        c.setFont('Helvetica', 12)
        c.drawString(inch, y_position, f"Total Cost: ${recommendation['cost']:.2f}")
        y_position -= 15
        c.drawString(inch, y_position, f"Remaining Budget: ${recommendation['remaining']:.2f}")
        y_position -= 30

        # Components List Header
        c.setFont('Helvetica-Bold', 14)
        c.drawString(inch, y_position, "Components:")
        y_position -= 20

        # Define column widths for a simple table-like layout
        col_width_type = 1.5 * inch
        col_width_name = 3.5 * inch
        col_width_price = 1 * inch
        col_width_specs = 2.5 * inch

        # Draw column headers
        c.setFont('Helvetica-Bold', 10)
        c.drawString(inch, y_position, "Type")
        c.drawString(inch + col_width_type, y_position, "Name")
        c.drawString(inch + col_width_type + col_width_name, y_position, "Price")
        c.drawString(inch + col_width_type + col_width_name + col_width_price, y_position, "Key Specifications")
        y_position -= 10
        c.line(inch, y_position, width - inch, y_position) # Separator line
        y_position -= 15

        c.setFont('Helvetica', 9) # Smaller font for component details

        # Iterate through each component in the build
        for part_type, part_data in recommendation['parts'].items():
            if part_type == '_ml_score': # Skip internal ML score
                continue

            # Check if new page is needed before drawing the next component
            if y_position < 70:
                c.showPage() # Start a new page
                # Redraw header on the new page
                c.setFillColorRGB(*self._hex_to_rgb(COLORS['secondary']))
                c.rect(0, height - 70, width, 70, fill=True)
                c.setFillColorRGB(*self._hex_to_rgb(COLORS['light']))
                c.setFont('Helvetica-Bold', 24)
                c.drawString(inch, height - 45, "AI PC Builder Pro - Build Recommendation (Cont.)")
                y_position = height - 100 # Reset Y position for new page content

                # Redraw column headers on the new page
                c.setFont('Helvetica-Bold', 10)
                c.drawString(inch, y_position, "Type")
                c.drawString(inch + col_width_type, y_position, "Name")
                c.drawString(inch + col_width_type + col_width_name, y_position, "Price")
                c.drawString(inch + col_width_type + col_width_name + col_width_price, y_position, "Key Specifications")
                y_position -= 10
                c.line(inch, y_position, width - inch, y_position)
                y_position -= 15
                c.setFont('Helvetica', 9)

            # Gather specifications for the current part based on SPEC_MAP
            specs_to_display = []
            if part_type in SPEC_MAP:
                for spec_key in SPEC_MAP[part_type]:
                    value = part_data.get(spec_key)
                    if value is not None and str(value) != 'nan':
                        display_key = ' '.join(word.capitalize() for word in spec_key.split('_'))
                        specs_to_display.append(f"{display_key}: {value}")
            elif part_type == "GPU": # Special handling for GPU VRAM/power if not in SPEC_MAP
                if part_data.get('VRAM') is not None and str(part_data.get('VRAM')) != 'nan':
                     specs_to_display.append(f"VRAM: {part_data['VRAM']}")
                if part_data.get('power') is not None and str(part_data.get('power')) != 'nan':
                     specs_to_display.append(f"Power: {part_data['power']}W")
            else:
                specs_to_display.append("No specific features defined for display.")

            spec_text = " | ".join(specs_to_display)

            # Draw part details
            c.drawString(inch, y_position, part_type)
            c.drawString(inch + col_width_type, y_position, part_data['name'])
            c.drawString(inch + col_width_type + col_width_name, y_position, f"${part_data['price']:.2f}")

            # Handle wrapping for specifications text
            textobject = c.beginText()
            textobject.setFont('Helvetica', 8)
            textobject.setFillColorRGB(*self._hex_to_rgb(COLORS['dark']))
            textobject.setTextOrigin(inch + col_width_type + col_width_name + col_width_price, y_position)
            for line in self._wrap_text(spec_text, 40): # Wrap specs if too long
                textobject.textLine(line)
            c.drawText(textobject)

            # Adjust Y position for the next component based on content height
            y_position -= max(15, len(self._wrap_text(spec_text, 40)) * 10) # Estimate 10 pts per line
            y_position -= 5 # Small padding between items

        # Footer for the PDF
        c.setFillColorRGB(*self._hex_to_rgb(COLORS['medium']))
        c.setFont('Helvetica-Oblique', 8)
        c.drawString(inch, 30, "Generated by AI PC Builder Pro")

        try:
            c.save() # Save the PDF file
            messagebox.showinfo("Export Successful", f"Build details exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not save PDF: {e}")

    def load_image(self, url, size=IMAGE_SIZE):
        """
        Loads an image from a URL, resizes it, and caches it.
        Returns a ImageTk.PhotoImage object.
        """
        if not url or not isinstance(url, str):
            return None
        cache_key = (url, size)
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            image = Image.open(io.BytesIO(response.content))
            image = image.resize(size, Image.LANCZOS) # High-quality resizing
            photo = ImageTk.PhotoImage(image)
            self.image_cache[cache_key] = photo
            return photo
        except requests.exceptions.RequestException as e:
            print(f"Error loading image from {url}: {e}")
            return None
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def create_smart_recommendation_tab_content(self, parent_frame):
        """
        Sets up the UI for generating smart recommendations within the smart_recommendation_tab.
        This includes input for scenario and a display area for the generated build.
        """
        for widget in parent_frame.winfo_children():
            widget.destroy() # Clear previous content

        controls_frame = tk.Frame(parent_frame, bg=COLORS['light'])
        controls_frame.pack(fill='x', padx=30, pady=20)

        tk.Label(controls_frame, text="🧠 AI Smart Recommendation", font=('Segoe UI', 18, 'bold'), bg=COLORS['light'], fg=COLORS['dark']).pack(anchor='w', pady=(0, 10))

        # Scenario selection
        scenario_frame = tk.Frame(controls_frame, bg=COLORS['light'])
        scenario_frame.pack(fill='x', pady=5)
        tk.Label(scenario_frame, text="Scenario:", font=('Segoe UI', 10, 'bold'), bg=COLORS['light'], fg=COLORS['dark']).pack(side='left', padx=(0, 10))
        
        self.scenario_var = tk.StringVar(value=self.backend.get_scenarios()[0]) # Default to first scenario from backend
        scenario_combo = ttk.Combobox(scenario_frame, textvariable=self.scenario_var, values=self.backend.get_scenarios(), state='readonly', font=('Segoe UI', 10), style='Modern.TCombobox')
        scenario_combo.pack(side='left', fill='x', expand=True)

        # Budget input (reusing existing budget_var from sidebar)
        budget_input_frame = tk.Frame(controls_frame, bg=COLORS['light'])
        budget_input_frame.pack(fill='x', pady=5)
        tk.Label(budget_input_frame, text="Max Budget ($):", font=('Segoe UI', 10, 'bold'), bg=COLORS['light'], fg=COLORS['dark']).pack(side='left', padx=(0, 10))
        tk.Entry(budget_input_frame, textvariable=self.budget_var, font=('Segoe UI', 10), bg=COLORS['light'], fg=COLORS['dark'], relief='flat', bd=0, insertbackground=COLORS['dark']).pack(side='left', fill='x', expand=True)

        ModernButton(controls_frame, text="✨ Generate Smart Builds", command=self.on_generate_smart_clicked, bg_color=COLORS['highlight'], hover_color=COLORS['accent']).pack(pady=(20, 10))

        # Container for displaying smart recommendations (scrollable)
        container = tk.Frame(parent_frame, bg=COLORS['light'])
        container.pack(fill='both', expand=True, padx=30, pady=(0, 20))
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.smart_recommendation_display_canvas = tk.Canvas(container, bg=COLORS['light'], highlightthickness=0)
        
        self.smart_recommendation_display_scrollbar_y = ttk.Scrollbar(container, orient="vertical", command=self.smart_recommendation_display_canvas.yview)
        self.smart_recommendation_display_scrollbar_x = ttk.Scrollbar(container, orient="horizontal", command=self.smart_recommendation_display_canvas.xview)
        
        self.smart_recommendation_display_canvas.configure(
            yscrollcommand=self.smart_recommendation_display_scrollbar_y.set,
            xscrollcommand=self.smart_recommendation_display_scrollbar_x.set
        )

        self.smart_recommendation_display_canvas.grid(row=0, column=0, sticky='nsew')
        self.smart_recommendation_display_scrollbar_y.grid(row=0, column=1, sticky='ns')
        self.smart_recommendation_display_scrollbar_x.grid(row=1, column=0, sticky='ew')
        
        self.smart_recommendation_scrollable_frame = tk.Frame(self.smart_recommendation_display_canvas, bg=COLORS['light'])
        self.smart_recommendation_display_canvas.create_window((0, 0), window=self.smart_recommendation_scrollable_frame, anchor="nw")

        self.smart_recommendation_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.smart_recommendation_display_canvas.configure(
                scrollregion=self.smart_recommendation_display_canvas.bbox("all")
            )
        )
        
        # Initial message for the smart recommendation tab
        self.smart_initial_message = tk.Label(self.smart_recommendation_scrollable_frame,
                 text="Select a scenario and budget, then click 'Generate Smart Builds' to see recommendations.",
                 font=('Segoe UI', 12, 'italic'), fg=COLORS['medium'], bg=COLORS['light'], wraplength=500)
        self.smart_initial_message.pack(pady=50, padx=20)
        
        # Bind mouse wheel for scrolling
        def _on_mousewheel_smart(event):
            self.smart_recommendation_display_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        def _on_shift_mousewheel_smart(event):
            self.smart_recommendation_display_canvas.xview_scroll(int(-1*(event.delta/120)), "units")

        self.smart_recommendation_display_canvas.bind_all("<MouseWheel>", _on_mousewheel_smart)
        self.smart_recommendation_display_canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel_smart)

    def on_generate_smart_clicked(self):
        """
        Initiates the smart build generation in a separate thread to keep the GUI responsive.
        """
        import threading # Local import for thread to avoid global import if not always used

        try:
            budget = float(self.budget_var.get())
            if budget <= 0:
                messagebox.showerror("Invalid Input", "Budget must be a positive number for smart recommendation.")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric budget for smart recommendation.")
            return

        # Clear existing content and show loading message
        for widget in self.smart_recommendation_scrollable_frame.winfo_children():
            widget.destroy()
        
        loading_label = tk.Label(self.smart_recommendation_scrollable_frame, text="🔄 Generating Smart Builds...", font=('Segoe UI', 16, 'bold'), bg=COLORS['light'], fg=COLORS['dark'])
        loading_label.pack(pady=50)
        self.root.update_idletasks() # Update GUI immediately

        # Start recommendation process in a new thread
        threading.Thread(target=self._run_smart_recommendation).start()

    def _run_smart_recommendation(self):
        """
        Executes the backend calls for smart recommendations (rule-based and ML-based).
        This method runs in a separate thread.
        """
        budget = float(self.budget_var.get())
        scenario = self.scenario_var.get()

        # Get rule-based smart build from backend
        rule_based_build = self.backend.generate_smart_recommendation_rule_based(scenario, budget)
        ml_based_builds = []
        
        # Attempt to get ML-based builds if a model exists for the scenario
        if scenario in self.models and self.models[scenario] is not None:
            try:
                ml_based_builds = self.backend.generate_ml_based_recommendations(budget, scenario=scenario, num_samples=50, top_n=3)
            except Exception as e:
                print(f"Error generating ML-based recommendation: {e}")
                # Use root.after to safely show messagebox from non-GUI thread
                self.root.after(0, lambda: messagebox.showwarning("ML Build Warning", f"Could not generate ML-based builds: {e}"))
        else:
            self.root.after(0, lambda: messagebox.showwarning("ML Model Missing", f"ML model for scenario '{scenario}' not found. Only rule-based recommendation will be available."))

        # After recommendations are generated, update the GUI (must be done on main thread)
        self.root.after(0, lambda: self._complete_smart_recommendation_display(rule_based_build, ml_based_builds, scenario, budget))

    def _complete_smart_recommendation_display(self, rule_based_build, ml_based_builds, scenario, original_budget):
        """
        Updates the GUI with the results of the smart recommendation process.
        Displays rule-based build and top N ML-optimized builds side-by-side.
        """
        for widget in self.smart_recommendation_scrollable_frame.winfo_children():
            widget.destroy() # Clear loading message

        if rule_based_build or ml_based_builds:
            self._display_dual_smart_recommendations(rule_based_build, ml_based_builds, scenario, original_budget)
            messagebox.showinfo("Smart Builds Generated", f"AI Smart Builds for '{scenario}' created successfully!")
        else:
            tk.Label(self.smart_recommendation_scrollable_frame, text=f"Could not generate any smart builds for '{scenario}' with the given budget. Try adjusting criteria.", font=('Segoe UI', 12, 'italic'), fg=COLORS['danger'], bg=COLORS['light'], wraplength=500).pack(pady=50, padx=20)
            messagebox.showerror("Smart Build Failed", f"Unable to generate any smart builds for '{scenario}'.")

    def _display_dual_smart_recommendations(self, rule_based_build, ml_based_builds, scenario, original_budget):
        """
        Lays out the rule-based and ML-optimized build cards horizontally.
        """
        for widget in self.smart_recommendation_scrollable_frame.winfo_children():
            widget.destroy()

        horizontal_cards_frame = tk.Frame(self.smart_recommendation_scrollable_frame, bg=COLORS['light'])
        horizontal_cards_frame.pack(fill='both', expand=False, padx=10, pady=10)

        current_column = 0

        # Display Rule-Based Build Card
        if rule_based_build:
            rule_cost = sum(part['price'] for part in rule_based_build.values() if isinstance(part, dict))
            rule_remaining = original_budget - rule_cost
            
            rule_ml_score_text = "N/A"
            if scenario in self.models and self.models[scenario] is not None:
                try:
                    # Predict ML score for rule-based build for comparison
                    rule_ml_score = self.backend.predict_build_score(rule_based_build, scenario)
                    if rule_ml_score is not None:
                        rule_ml_score_text = f"{rule_ml_score:.2f}"
                    else:
                        rule_ml_score_text = "Failed"
                except Exception as e:
                    print(f"Error predicting ML score for rule-based build: {e}")
            
            rule_display_rec = {
                "parts": rule_based_build,
                "cost": rule_cost,
                "remaining": rule_remaining,
                "type": f"Rule-Based Build ({scenario})\n🤖 ML Score: {rule_ml_score_text}"
            }
            card_frame = tk.Frame(horizontal_cards_frame, bg=COLORS['light'], width=450, height=550)
            card_frame.grid(row=0, column=current_column, padx=10, pady=10, sticky="nsew")
            card_frame.grid_propagate(False) # Prevent card from resizing based on content
            self.create_build_card(card_frame, rule_display_rec, "RuleBased") # Index "RuleBased" is a string
            current_column += 1
        else:
            card_frame = tk.Frame(horizontal_cards_frame, bg=COLORS['light'], width=450, height=550)
            card_frame.grid(row=0, column=current_column, padx=10, pady=10, sticky="nsew")
            card_frame.grid_propagate(False)
            tk.Label(card_frame, text="No Rule-Based Build Generated", font=('Segoe UI', 14, 'italic'), fg=COLORS['medium'], bg=COLORS['light'], wraplength=300).pack(pady=50, padx=20, fill='both', expand=True)
            current_column += 1

        # Display ML-Optimized Build Cards
        if ml_based_builds:
            for i, ml_build in enumerate(ml_based_builds):
                ml_cost = sum(part['price'] for part in ml_build.values() if isinstance(part, dict))
                ml_remaining = original_budget - ml_cost
                
                ml_score = ml_build.get('_ml_score', 'N/A')
                ml_score_text = f"{ml_score:.2f}" if isinstance(ml_score, (int, float)) else str(ml_score) # Ensure score is formatted
                
                ml_display_rec = {
                    "parts": ml_build,
                    "cost": ml_cost,
                    "remaining": ml_remaining,
                    "type": f"ML-Optimized Build {i+1} ({scenario})\n🤖 ML Score: {ml_score_text}"
                }
                card_frame = tk.Frame(horizontal_cards_frame, bg=COLORS['light'], width=450, height=550)
                card_frame.grid(row=0, column=current_column, padx=10, pady=10, sticky="nsew")
                card_frame.grid_propagate(False)
                self.create_build_card(card_frame, ml_display_rec, f"MLBased{i+1}") # Index string for ML builds
                current_column += 1
        else:
            card_frame = tk.Frame(horizontal_cards_frame, bg=COLORS['light'], width=450, height=550)
            card_frame.grid(row=0, column=current_column, padx=10, pady=10, sticky="nsew")
            card_frame.grid_propagate(False)
            tk.Label(card_frame, text="No ML-Optimized Builds Generated (ML Model for scenario may be missing or failed)", font=('Segoe UI', 14, 'italic'), fg=COLORS['medium'], bg=COLORS['light'], wraplength=300).pack(pady=50, padx=20, fill='both', expand=True)
            current_column += 1

        # Configure columns to distribute space
        for c in range(current_column):
            horizontal_cards_frame.grid_columnconfigure(c, weight=1)

        # Update scroll region after all content is packed
        self.smart_recommendation_scrollable_frame.update_idletasks()
        self.smart_recommendation_display_canvas.config(scrollregion=self.smart_recommendation_display_canvas.bbox("all"))

    def create_custom_build_tab_content(self, parent_frame):
        """
        Sets up the UI for the 'Custom Build' tab, allowing users to manually
        select each component using dropdowns.
        """
        self.custom_parts = {} # Stores Tkinter StringVars for each part's selection
        top = tk.Frame(parent_frame, bg=COLORS['light'])
        top.pack(pady=10)

        row = 0
        for part_type, df in self.parts_data.items(): # Iterate through all available part types
            tk.Label(top, text=f"Select {part_type}:", bg=COLORS['light'], fg=COLORS['dark'], font=('Segoe UI', 11)).grid(row=row, column=0, padx=10, pady=5, sticky='w')
            options = df['name'].tolist() # Get list of part names for dropdown
            var = tk.StringVar()
            menu = ttk.Combobox(top, textvariable=var, values=options, width=50, state='readonly')
            menu.grid(row=row, column=1, padx=10, pady=5)
            self.custom_parts[part_type] = var # Store StringVar for later retrieval
            row += 1

        tk.Button(top, text="View Build", command=self.show_custom_build,
                     bg=COLORS['success'], fg='white', font=('Segoe UI', 10, 'bold')).grid(row=row, columnspan=2, pady=10)

        self.custom_result = tk.Frame(parent_frame, bg=COLORS['light'])
        self.custom_result.pack(fill='both', expand=True, padx=20, pady=10)

    def show_custom_build(self):
        """
        Displays the details and total cost of the components selected in the 'Custom Build' tab.
        """
        for widget in self.custom_result.winfo_children():
            widget.destroy() # Clear previous results

        total_price = 0
        tk.Label(self.custom_result, text="Your Custom Build:", font=('Segoe UI', 14, 'bold'), bg=COLORS['light'], fg=COLORS['accent']).pack(anchor='w')

        # Iterate through selected parts and display their details
        for part_type, var in self.custom_parts.items():
            selected = var.get() # Get the selected part name
            df = self.parts_data[part_type]
            match = df[df['name'] == selected] # Find the full part data
            if not match.empty:
                row = match.iloc[0]
                price = float(row['price'])
                total_price += price
                info = f"{part_type}: {selected} (${price:.2f})"
                tk.Label(self.custom_result, text=info, bg=COLORS['light'], fg=COLORS['dark'], font=('Segoe UI', 11)).pack(anchor='w')

        tk.Label(self.custom_result, text=f"Total Price: ${total_price:.2f}", bg=COLORS['light'], fg=COLORS['accent'], font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(10, 0))

    def create_graphs_tab_content(self, parent_frame):
        """
        Sets up the UI for the 'Graphs' tab.
        Contains a dropdown for part type selection and a button to generate graphs in a new window.
        """
        # Clear any previous content from the tab
        for widget in parent_frame.winfo_children():
            widget.destroy()

        top = tk.Frame(parent_frame, bg=COLORS['light'])
        top.pack(pady=20, padx=20)

        tk.Label(top, text="Select Part Type for Graphs:", bg=COLORS['light'], fg=COLORS['dark'], font=('Segoe UI', 11)).pack(side='left')
        
        # Use FEATURE_COLUMNS_FOR_GRAPHS keys as options for plotting
        self.graph_part_type = tk.StringVar(value=list(FEATURE_COLUMNS_FOR_GRAPHS.keys())[0])
        menu = ttk.Combobox(top, textvariable=self.graph_part_type, values=list(FEATURE_COLUMNS_FOR_GRAPHS.keys()), state='readonly', style='Modern.TCombobox', width=30)
        menu.pack(side='left', padx=10)

        # Button to generate graphs (will open a new Toplevel window)
        ModernButton(top, text="📊 Generate Graphs", command=self.draw_graphs,
                     bg_color=COLORS['accent'], hover_color=COLORS['highlight']).pack(side='left', padx=10)

        # Optional: Add a placeholder message or image for the tab itself
        tk.Label(parent_frame, text="Graphs will open in a separate window.",
                 font=('Segoe UI', 12, 'italic'), fg=COLORS['medium'], bg=COLORS['light']).pack(pady=50)


    def draw_graphs(self):
        """
        Generates and displays two types of graphs (bar chart and radar chart)
        for the selected part type in a new Toplevel window.
        """
        # Close any existing graph window before opening a new one
        if self.graph_window and self.graph_window.winfo_exists():
            self.graph_window.destroy()
            self.graph_window = None # Clear the reference

        part_type = self.graph_part_type.get()
        df = self.parts_data.get(part_type)
        features = FEATURE_COLUMNS_FOR_GRAPHS.get(part_type, [])

        if df is None or len(df) < 3:
            messagebox.showinfo("Insufficient Data", "Not enough data to draw graphs (need at least 3 items for meaningful charts).")
            return

        self.graph_window = tk.Toplevel(self.root)
        self.graph_window.title(f"{part_type} Performance Graphs")
        self.graph_window.geometry("1400x700") # Adjust size for two charts side-by-side
        self.graph_window.configure(bg=COLORS['light'])
        self.graph_window.transient(self.root) # Make it modal
        self.graph_window.grab_set()

        # Set protocol for closing the window to ensure matplotlib figures are closed
        self.graph_window.protocol("WM_DELETE_WINDOW", self._on_graph_window_close)

        # Main frame inside the graph window for layout
        main_graph_frame = tk.Frame(self.graph_window, bg=COLORS['light'])
        main_graph_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Frame for charts (horizontal layout)
        charts_frame = tk.Frame(main_graph_frame, bg=COLORS['light'])
        charts_frame.pack(fill='both', expand=True)

        # --- Bar Chart: Top 5 Cheapest Components ---
        fig1, ax1 = plt.subplots(figsize=(6, 5), facecolor=COLORS['light']) # Adjusted figsize
        fig1.patch.set_facecolor(COLORS['light'])
        ax1.set_facecolor(COLORS['card'])

        df_sorted = df.sort_values(by='price').head(5)
        ax1.bar(df_sorted['name'], df_sorted['price'], color=COLORS['success'])
        ax1.set_title(f"Top 5 Cheapest {part_type}s", color=COLORS['dark'], fontsize=14)
        ax1.set_ylabel("Price ($)", color=COLORS['dark'])
        ax1.tick_params(axis='x', rotation=45, labelsize=9, colors=COLORS['dark'])
        ax1.tick_params(axis='y', labelsize=9, colors=COLORS['dark'])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(COLORS['dark'])
        ax1.spines['left'].set_color(COLORS['dark'])
        plt.tight_layout()

        canvas1 = FigureCanvasTkAgg(fig1, master=charts_frame)
        canvas1.draw()
        canvas_widget1 = canvas1.get_tk_widget()
        canvas_widget1.pack(side='left', padx=10, fill='both', expand=True)

        toolbar1 = NavigationToolbar2Tk(canvas1, charts_frame)
        toolbar1.update()
        toolbar1.pack(side='left', padx=(0, 10)) # Pack toolbar next to first canvas

        # --- Radar Chart: Feature comparison for a sample component ---
        if features and len(df_sorted) > 0:
            fig2 = plt.figure(figsize=(6, 5), facecolor=COLORS['light']) # Adjusted figsize
            ax2 = fig2.add_subplot(111, polar=True)
            fig2.patch.set_facecolor(COLORS['light'])
            ax2.set_facecolor(COLORS['card'])

            sample = df_sorted.iloc[0] # Take the cheapest as a sample for radar chart
            
            normalized_values = []
            actual_feature_labels = []

            for feature in features:
                val = sample.get(feature)
                if pd.api.types.is_numeric_dtype(type(val)) and pd.notna(val):
                    full_feature_name = f"{part_type}_{feature}"
                    if full_feature_name in self.normalization_ranges:
                        min_val = self.normalization_ranges[full_feature_name]['min']
                        max_val = self.normalization_ranges[full_feature_name]['max']
                        if max_val - min_val > 0:
                            normalized_val = (val - min_val) / (max_val - min_val)
                        else:
                            normalized_val = 0.5 # Default if all values are the same
                        normalized_values.append(normalized_val * 10) # Scale to 0-10
                        actual_feature_labels.append(feature.replace('_', ' ').title()) # Human-readable label
                    else:
                        print(f"Skipping radar feature '{feature}' for {part_type}: No normalization range.")
                else:
                    print(f"Skipping radar feature '{feature}' for {part_type}: Not numeric or NaN.")

            if len(actual_feature_labels) < 2:
                tk.Label(charts_frame, text=f"Not enough numerical features for a radar chart for {part_type}. (Need at least 2, found {len(actual_feature_labels)})", bg=COLORS['light'], fg=COLORS['danger'], wraplength=300).pack(side='left', padx=10, fill='both', expand=True)
            else:
                angles = [n / float(len(actual_feature_labels)) * 2 * np.pi for n in range(len(actual_feature_labels))]
                values_closed = normalized_values + normalized_values[:1]
                angles_closed = angles + angles[:1]

                ax2.plot(angles_closed, values_closed, 'o-', linewidth=2, color=COLORS['highlight'])
                ax2.fill(angles_closed, values_closed, color=COLORS['highlight'], alpha=0.25)
                ax2.set_thetagrids([a * 180 / np.pi for a in angles], actual_feature_labels, color=COLORS['dark'], fontsize=10)
                ax2.set_yticks([2, 4, 6, 8, 10])
                ax2.set_yticklabels([str(x) for x in [2, 4, 6, 8, 10]], color=COLORS['dark'], fontsize=9)
                ax2.set_ylim(0, 10)
                ax2.grid(True, color=COLORS['grid'], alpha=0.7)
                ax2.set_title(f"Feature Radar for {sample['name']}", color=COLORS['dark'], fontsize=14) # Title at the end
                plt.tight_layout()

                canvas2 = FigureCanvasTkAgg(fig2, master=charts_frame)
                canvas2.draw()
                canvas_widget2 = canvas2.get_tk_widget()
                canvas_widget2.pack(side='left', padx=10, fill='both', expand=True)

                toolbar2 = NavigationToolbar2Tk(canvas2, charts_frame)
                toolbar2.update()
                toolbar2.pack(side='left', padx=(0, 10)) # Pack toolbar next to second canvas
        else:
            tk.Label(charts_frame, text=f"No suitable data for radar chart for {part_type}.", bg=COLORS['light'], fg=COLORS['danger'], wraplength=300).pack(side='left', padx=10, fill='both', expand=True)

        # Close all figures explicitly when the graph window itself is closed
        # Handled by _on_graph_window_close protocol


    def _on_graph_window_close(self):
        """Handles the closing of the graph window, ensuring Matplotlib figures are closed."""
        if self.graph_window:
            self.graph_window.destroy()
            self.graph_window = None
            plt.close('all') # Crucial: Close all Matplotlib figures associated with this window


    def create_compare_tab_content(self, parent_frame):
        """
        Sets up the UI for the 'Component Comparison' tab.
        Allows selecting two components of the same type for detailed comparison.
        The comparison results will open in a separate Toplevel window.
        """
        # Clear any previous content from the tab
        for widget in parent_frame.winfo_children():
            widget.destroy()

        header_frame = tk.Frame(parent_frame, bg=COLORS['light'])
        header_frame.pack(fill='x', pady=(10, 0), padx=10)

        tk.Label(header_frame, text="Compare Two Components", font=('Segoe UI', 18, 'bold'), bg=COLORS['light'], fg=COLORS['dark']).pack(anchor='w', pady=(0, 10))

        # Dropdowns for selecting components
        selection_frame = ModernCard(parent_frame, bg_color=COLORS['card'])
        selection_frame.pack(fill='x', pady=(0, 10), padx=10)

        # Part Type selection
        part_type_frame = tk.Frame(selection_frame, bg=COLORS['card'])
        part_type_frame.pack(fill='x', padx=15, pady=5)
        tk.Label(part_type_frame, text="Select Part Type:", bg=COLORS['card'], fg=COLORS['dark'], font=('Segoe UI', 11)).pack(side='left', padx=(0, 10))
        self.compare_type = tk.StringVar(value=list(self.parts_data.keys())[0]) # Default to first part type
        compare_menu = ttk.Combobox(part_type_frame, textvariable=self.compare_type, values=list(self.parts_data.keys()), state='readonly', style='Modern.TCombobox', width=40)
        compare_menu.pack(side='left', fill='x', expand=True)
        compare_menu.bind("<<ComboboxSelected>>", lambda e: self.update_compare_options())

        # Component 1 selection
        part1_frame = tk.Frame(selection_frame, bg=COLORS['card'])
        part1_frame.pack(fill='x', padx=15, pady=5)
        tk.Label(part1_frame, text="Component 1:", bg=COLORS['card'], fg=COLORS['dark'], font=('Segoe UI', 11)).pack(side='left', padx=(0, 10))
        self.part1_var = tk.StringVar()
        self.part1_menu = ttk.Combobox(part1_frame, textvariable=self.part1_var, state='readonly', style='Modern.TCombobox', width=40)
        self.part1_menu.pack(side='left', fill='x', expand=True)

        # Component 2 selection
        part2_frame = tk.Frame(selection_frame, bg=COLORS['card'])
        part2_frame.pack(fill='x', padx=15, pady=5)
        tk.Label(part2_frame, text="Component 2:", bg=COLORS['card'], fg=COLORS['dark'], font=('Segoe UI', 11)).pack(side='left', padx=(0, 10))
        self.part2_var = tk.StringVar()
        self.part2_menu = ttk.Combobox(part2_frame, textvariable=self.part2_var, state='readonly', style='Modern.TCombobox', width=40)
        self.part2_menu.pack(side='left', fill='x', expand=True)

        # Compare button
        compare_button_frame = tk.Frame(selection_frame, bg=COLORS['card'])
        compare_button_frame.pack(pady=15)
        ModernButton(compare_button_frame, text="  Compare Components  ", command=self.compare_parts,
                     bg_color=COLORS['highlight'], hover_color=COLORS['accent']).pack()

        # Initial population of component dropdowns
        self.update_compare_options()
        
        # Optional: Add a placeholder message for the tab itself
        tk.Label(parent_frame, text="Comparison results will open in a separate window.",
                 font=('Segoe UI', 12, 'italic'), fg=COLORS['medium'], bg=COLORS['light']).pack(pady=50)


    def update_compare_options(self):
        """
        Updates the available options in the component dropdowns based on the selected part type.
        """
        part_type = self.compare_type.get()
        options = self.parts_data[part_type]['name'].tolist() # Get all names for selected part type
        self.part1_menu['values'] = options
        self.part2_menu['values'] = options
        if options:
            self.part1_var.set(options[0]) # Default first dropdown to first item
            self.part2_var.set(options[1 if len(options) > 1 else 0]) # Default second to second, or first if only one
        # No need to clear content here, as it's in a separate window now
        # self.clear_comparison_display_content() 

    def compare_parts(self):
        """
        Performs the comparison between two selected components and displays results
        in a new Toplevel window.
        """
        # Close any existing comparison window before opening a new one
        if self.compare_window and self.compare_window.winfo_exists():
            self.compare_window.destroy()
            self.compare_window = None # Clear the reference

        part_type = self.compare_type.get()
        df = self.parts_data[part_type]
        name1 = self.part1_var.get()
        name2 = self.part2_var.get()

        if not name1 or not name2:
            messagebox.showerror("Selection Error", "Please select two components for comparison.")
            return

        # Retrieve full data for the selected components
        row1 = df[df['name'] == name1].iloc[0].to_dict()
        row2 = df[df['name'] == name2].iloc[0].to_dict()

        if row1['name'] == row2['name'] and row1['price'] == row2['price']:
            messagebox.showwarning("Selection Warning", "Please select two *different* components for comparison.")
            return

        self.compare_window = tk.Toplevel(self.root)
        self.compare_window.title(f"Component Comparison: {name1} vs {name2}")
        self.compare_window.geometry("1400x900") # Adjust size for pop-up
        self.compare_window.configure(bg=COLORS['light'])
        self.compare_window.transient(self.root) # Make it modal to the main window
        self.compare_window.grab_set()

        # Set protocol for closing the window to ensure matplotlib figures are closed
        self.compare_window.protocol("WM_DELETE_WINDOW", self._on_compare_window_close)

        # Create a scrollable area within the comparison window
        compare_scroll_canvas = tk.Canvas(self.compare_window, bg=COLORS['light'], highlightthickness=0)
        compare_scroll_canvas.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        compare_scroll_v_scrollbar = ttk.Scrollbar(self.compare_window, orient='vertical', command=compare_scroll_canvas.yview)
        compare_scroll_v_scrollbar.pack(side='right', fill='y')
        compare_scroll_h_scrollbar = ttk.Scrollbar(self.compare_window, orient='horizontal', command=compare_scroll_canvas.xview)
        compare_scroll_h_scrollbar.pack(side='bottom', fill='x')

        compare_scroll_canvas.configure(yscrollcommand=compare_scroll_v_scrollbar.set,
                                        xscrollcommand=compare_scroll_h_scrollbar.set)

        compare_results_inner_frame = tk.Frame(compare_scroll_canvas, bg=COLORS['light'])
        compare_scroll_canvas.create_window((0, 0), window=compare_results_inner_frame, anchor='nw')

        compare_results_inner_frame.bind(
            "<Configure>",
            lambda e: compare_scroll_canvas.configure(
                scrollregion=compare_scroll_canvas.bbox("all")
            )
        )
        compare_scroll_canvas.bind_all("<MouseWheel>", lambda e: compare_scroll_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        compare_scroll_canvas.bind_all("<Shift-MouseWheel>", lambda e: compare_scroll_canvas.xview_scroll(int(-1*(e.delta/120)), "units"))


        # "Close Comparison" button (top right of comparison results area)
        close_comparison_btn = ModernButton(compare_results_inner_frame, text="  + Close Comparison  ",
                                                 command=self._on_compare_window_close, # Call the close handler
                                                 bg_color=COLORS['danger'], hover_color=COLORS['dark'])
        close_comparison_btn.pack(side='top', anchor='ne', pady=10, padx=10)

        # Container for component images
        comp_image_frame = tk.Frame(compare_results_inner_frame, bg=COLORS['light'])
        comp_image_frame.pack(fill="x", pady=5)
        comp1_img_label = tk.Label(comp_image_frame, bg=COLORS['light'])
        comp2_img_label = tk.Label(comp_image_frame, bg=COLORS['light'])
        comp1_img_label.pack(side="left", expand=True, padx=10)
        comp2_img_label.pack(side="right", expand=True, padx=10)

        # Populate and display Component Images
        img1 = self.load_image(row1.get('image'), size=(100, 100))
        img2 = self.load_image(row2.get('image'), size=(100, 100))
        if img1:
            comp1_img_label.config(image=img1)
            comp1_img_label.image = img1
        else:
            placeholder_url = f"https://placehold.co/100x100/6c757d/f8f9fa?text={row1['name'].replace(' ', '%20')}"
            img1_placeholder = self.load_image(placeholder_url, size=(100,100))
            comp1_img_label.config(image=img1_placeholder)
            comp1_img_label.image = img1_placeholder

        if img2:
            comp2_img_label.config(image=img2)
            comp2_img_label.image = img2
        else:
            placeholder_url = f"https://placehold.co/100x100/6c757d/f8f9fa?text={row2['name'].replace(' ', '%20')}"
            img2_placeholder = self.load_image(placeholder_url, size=(100,100))
            comp2_img_label.config(image=img2_placeholder)
            comp2_img_label.image = img2_placeholder


        # Comparison table (Treeview)
        comparison_table_container = ModernCard(compare_results_inner_frame, bg_color=COLORS['card'])
        comparison_table_container.pack(fill="x", pady=10)
        
        # Create Treeview and its scrollbars
        comparison_tree = ttk.Treeview(comparison_table_container, show="headings", height=8, style="Modern.Treeview")
        comparison_tree_h_scrollbar = ttk.Scrollbar(comparison_table_container, orient="horizontal", command=comparison_tree.xview)
        comparison_tree_v_scrollbar = ttk.Scrollbar(comparison_table_container, orient="vertical", command=comparison_tree.yview)
        comparison_tree.configure(xscrollcommand=comparison_tree_h_scrollbar.set, yscrollcommand=comparison_tree_v_scrollbar.set)

        comparison_tree.pack(fill="both", expand=True, padx=15, pady=15)
        comparison_tree_h_scrollbar.pack(side="bottom", fill="x")
        comparison_tree_v_scrollbar.pack(side="right", fill="y")
        
        self.populate_comparison_table_in_window(comparison_tree, part_type, row1, row2)


        # Radar chart container
        radar_chart_container = ModernCard(compare_results_inner_frame, bg_color=COLORS['card'])
        radar_chart_container.pack(fill="both", expand=True, pady=10)
        self.generate_radar_chart_in_window(radar_chart_container, part_type, row1, row2)

        # Update the scrollregion of the comparison canvas after all content is packed
        compare_results_inner_frame.update_idletasks()
        compare_scroll_canvas.config(scrollregion=compare_scroll_canvas.bbox("all"))

    def _on_compare_window_close(self):
        """Handles the closing of the comparison window, ensuring Matplotlib figures are closed."""
        if self.compare_window:
            self.compare_window.destroy()
            self.compare_window = None
            plt.close('all') # Close all Matplotlib figures associated with this window


    def populate_comparison_table_in_window(self, tree_widget, part_type, comp1_data, comp2_data):
        """
        Populates the given Treeview table with feature-by-feature comparison of two components.
        This version takes the tree_widget as an argument, allowing it to be used in a Toplevel.
        """
        for item in tree_widget.get_children(): # Clear existing rows
            tree_widget.delete(item)

        ordered_features = ['name', 'price']
        if part_type in SPEC_MAP:
            for spec in SPEC_MAP[part_type]:
                if spec not in ordered_features:
                    ordered_features.append(spec)
        all_keys_set = set(comp1_data.keys()).union(set(comp2_data.keys()))
        other_cols = sorted(list(all_keys_set - set(ordered_features) - {'image', 'url', 'partType', 'brand', 'socket', 'size', 'type', '_ml_score', 'score'}))
        display_cols = ordered_features + other_cols

        columns = ["Feature", comp1_data.get('name', 'Component 1'), comp2_data.get('name', 'Component 2')]
        tree_widget["columns"] = columns
        
        for col_idx, col_name in enumerate(columns):
            display_name = col_name
            if col_idx > 0 and len(display_name) > 30:
                display_name = display_name[:27] + "..."
            tree_widget.heading(col_name, text=display_name.replace('_', ' ').title(), anchor="w")
            tree_widget.column(col_name, width=150, minwidth=100, stretch=tk.YES)

        for feature_key in display_cols:
            val1 = comp1_data.get(feature_key, 'N/A')
            val2 = comp2_data.get(feature_key, 'N/A')
            
            display_feature_name = feature_key.replace('_', ' ').title()
            
            tree_widget.insert("", "end", values=(display_feature_name, val1, val2))
        
        for i, col_name in enumerate(columns):
            max_width = len(col_name) * 10
            for item_id in tree_widget.get_children():
                value = tree_widget.item(item_id, 'values')[i]
                content_width = len(str(value)) * 9
                if content_width > max_width:
                    max_width = content_width
            tree_widget.column(col_name, width=max(100, max_width + 20))


    def generate_radar_chart_in_window(self, parent_container, part_type, comp1_data, comp2_data):
        """
        Generates and displays an enhanced radar chart comparing two components' key features.
        Features are normalized using precomputed ranges. This version plots into a given parent_container.
        """
        # No need to clear self.radar_canvas etc. here, as they are local to the new window scope
        # and will be garbage collected with the Toplevel.
        
        features_to_plot = FEATURE_COLUMNS_FOR_GRAPHS.get(part_type, [])
        
        if not features_to_plot:
            no_data_label = tk.Label(parent_container,
                                     text=f"No comparison data available for {part_type}",
                                     font=('Arial', 12), fg=COLORS['dark'], bg=COLORS['card'])
            no_data_label.pack(expand=True, fill=tk.BOTH)
            return
        
        comp1_values = []
        comp2_values = []
        actual_feature_labels = []

        for feature in features_to_plot:
            val1 = comp1_data.get(feature)
            val2 = comp2_data.get(feature)

            try:
                val1 = float(val1) if pd.notna(val1) else np.nan
            except (ValueError, TypeError):
                val1 = np.nan
            try:
                val2 = float(val2) if pd.notna(val2) else np.nan
            except (ValueError, TypeError):
                val2 = np.nan

            full_feature_name = f"{part_type}_{feature}"
            if not np.isnan(val1) and not np.isnan(val2) and full_feature_name in self.normalization_ranges:
                min_val = self.normalization_ranges[full_feature_name]['min']
                max_val = self.normalization_ranges[full_feature_name]['max']

                if max_val - min_val > 0:
                    normalized_val1 = (val1 - min_val) / (max_val - min_val)
                    normalized_val2 = (val2 - min_val) / (max_val - min_val)
                else:
                    normalized_val1 = 0.5
                    normalized_val2 = 0.5
                
                comp1_values.append(normalized_val1 * 10)
                comp2_values.append(normalized_val2 * 10)
                actual_feature_labels.append(feature.replace('_', ' ').title())
            else:
                print(f"Skipping feature '{feature}' for radar chart due to non-numeric/NaN values or missing normalization range for {part_type}.")
        
        if not actual_feature_labels or len(actual_feature_labels) < 2:
            tk.Label(parent_container,
                     text=f"Not enough comparable numerical features for a radar chart for {part_type}. (Need at least 2, found {len(actual_feature_labels)})",
                     font=('Arial', 12), fg=COLORS['dark'], bg=COLORS['card'], wraplength=400).pack(pady=20)
            return

        num_features = len(actual_feature_labels)
        angles = [n / float(num_features) * 2 * np.pi for n in range(num_features)]
        angles += angles[:1]
        
        comp1_values_closed = comp1_values + comp1_values[:1]
        comp2_values_closed = comp2_values + comp2_values[:1]
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor('white')
        
        grid_values = [2, 4, 6, 8, 10]
        ax.set_ylim(0, 10)
        ax.set_yticks(grid_values)
        ax.set_yticklabels([str(int(v)) for v in grid_values], fontsize=9,
                           color=COLORS['text'], alpha=0.7)
        ax.grid(True, color=COLORS['grid'], alpha=0.7, linewidth=0.8, linestyle='--')
        
        for radius in grid_values:
            circle = mpatches.Circle((0, 0), radius, transform=ax.transData,
                                     fill=False, color=COLORS['grid'],
                                     alpha=0.5, linewidth=0.7, linestyle=':')
            ax.add_patch(circle)

        ax.fill(angles, comp1_values_closed,
                color=COLORS['highlight'], alpha=0.3,
                label=comp1_data['name'])
        ax.plot(angles, comp1_values_closed,
                color=COLORS['highlight'], linewidth=3,
                marker='o', markersize=8, markerfacecolor='white',
                markeredgecolor=COLORS['highlight'], markeredgewidth=2)
        
        ax.fill(angles, comp2_values_closed,
                color=COLORS['success'], alpha=0.3,
                label=comp2_data['name'])
        ax.plot(angles, comp2_values_closed,
                color=COLORS['success'], linewidth=3,
                marker='s', markersize=8, markerfacecolor='white',
                markeredgecolor=COLORS['success'], markeredgewidth=2)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(actual_feature_labels, color=COLORS['text'],
                           size=11, fontweight='bold')
        
        for i, (angle, val1, val2) in enumerate(zip(angles[:-1], comp1_values, comp2_values)):
            ax.text(angle, val1 + 0.4, f'{val1:.1f}',
                    ha='center', va='center', fontsize=8,
                    color=COLORS['highlight'], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=COLORS['highlight'], alpha=0.9, linewidth=0.5))
            
            va2 = 'bottom' if val2 < 1.5 else 'top'
            ax.text(angle, val2 - 0.4 if val2 > 1.5 else val2 + 0.4, f'{val2:.1f}',
                    ha='center', va=va2, fontsize=8,
                    color=COLORS['success'], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=COLORS['success'], alpha=0.9, linewidth=0.5))
        
        ax.set_title(f"Performance Comparison\n{comp1_data['name']} vs {comp2_data['name']}",
                     color=COLORS['dark'], fontsize=14, fontweight='bold',
                     va='bottom', pad=30)
        
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
                            fontsize=12, frameon=True, fancybox=True,
                            shadow=True, framealpha=0.95)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor(COLORS['dark'])
        
        avg1 = sum(comp1_values) / len(comp1_values)
        avg2 = sum(comp2_values) / len(comp2_values)
        
        if avg1 > avg2:
            winner_text = f"🏆 {comp1_data['name']} leads"
            winner_color = COLORS['highlight']
        elif avg2 > avg1:
            winner_text = f"🏆 {comp2_data['name']} leads"
            winner_color = COLORS['success']
        else:
            winner_text = "🤝 Equal performance"
            winner_color = COLORS['dark']
        
        summary_text = f"Overall Scores:\n{comp1_data['name']}: {avg1:.1f}/10\n{comp2_data['name']}: {avg2:.1f}/10\n\n{winner_text}"
        
        ax.text(0.02, 0.98, summary_text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor=winner_color, alpha=0.95, linewidth=2),
                color=COLORS['dark'], fontweight='bold')
        
        strengths_text = self._analyze_component_strengths(comp1_values, comp2_values,
                                                          actual_feature_labels,
                                                          comp1_data['name'], comp2_data['name'])
        
        ax.text(0.98, 0.02, strengths_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'],
                          edgecolor=COLORS['dark'], alpha=0.9),
                color=COLORS['dark'])
        
        plt.tight_layout()

        chart_frame = tk.Frame(parent_container, bg=COLORS['card'])
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Store a reference to the canvas and toolbar associated with this specific plot
        # so they can be cleaned up later if needed, though closing the Toplevel handles it.
        radar_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas_widget = radar_canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        radar_toolbar_frame = tk.Frame(chart_frame, bg=COLORS['card'])
        radar_toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        radar_toolbar = NavigationToolbar2Tk(radar_canvas, radar_toolbar_frame)
        radar_toolbar.update()
        
        controls_frame = tk.Frame(radar_toolbar_frame, bg=COLORS['card'])
        controls_frame.pack(side=tk.RIGHT, padx=10)
        
        export_btn = tk.Button(controls_frame, text="📊 Export Chart",
                                command=lambda: self._export_radar_chart(fig),
                                bg=COLORS['highlight'], fg='white', font=('Arial', 9, 'bold'),
                                relief=tk.FLAT, padx=10, pady=2)
        export_btn.pack(side=tk.LEFT, padx=2)
        
        grid_btn = tk.Button(controls_frame, text="⚏ Toggle Grid",
                                command=lambda ax=ax, canvas=radar_canvas: self._toggle_radar_grid(ax, canvas), # Pass ax and canvas
                                bg=COLORS['success'], fg='white', font=('Arial', 9, 'bold'),
                                relief=tk.FLAT, padx=10, pady=2)
        grid_btn.pack(side=tk.LEFT, padx=2)
        
        fullscreen_btn = tk.Button(controls_frame, text="🔍 Fullscreen",
                                    command=lambda: self._show_fullscreen_chart(comp1_data, comp2_data, actual_feature_labels, comp1_values, comp2_values),
                                    bg=COLORS['dark'], fg='white', font=('Arial', 9, 'bold'),
                                    relief=tk.FLAT, padx=10, pady=2)
        fullscreen_btn.pack(side=tk.LEFT, padx=2)
        
        radar_canvas.draw()
        
        # IMPORTANT: Close the figure immediately after embedding it.
        # It's now handled by the _on_compare_window_close protocol.
        # plt.close(fig) 


    def _analyze_component_strengths(self, comp1_vals, comp2_vals, labels, name1, name2):
        """
        Analyzes and summarizes the key strengths of each component based on their
        normalized feature values.
        """
        comp1_strengths = []
        comp2_strengths = []
        
        # Compare each feature's value with a small tolerance
        for i, (val1, val2, label) in enumerate(zip(comp1_vals, comp2_vals, labels)):
            if val1 > val2 + 0.5: # Comp1 significantly better
                comp1_strengths.append(label)
            elif val2 > val1 + 0.5: # Comp2 significantly better
                comp2_strengths.append(label)
        
        strengths_text = "Key Strengths:\n"
        if comp1_strengths:
            strengths_text += f"• {name1}: {', '.join(comp1_strengths[:2])}" # Show top 2 strengths
            if len(comp1_strengths) > 2:
                strengths_text += "..." # Indicate more if present
            strengths_text += "\n"
        if comp2_strengths:
            strengths_text += f"• {name2}: {', '.join(comp2_strengths[:2])}"
            if len(comp2_strengths) > 2:
                strengths_text += "..."
            strengths_text += "\n"
        
        if not comp1_strengths and not comp2_strengths:
            strengths_text += "Very close performance across all metrics."
        
        return strengths_text.strip()

    def _export_radar_chart(self, fig):
        """Allows the user to export the radar chart to an image file."""
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Radar Chart",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            if filename:
                # Set figure background color for export based on file type
                fig_facecolor = 'white'
                if filename.lower().endswith(('.pdf', '.svg')):
                    fig_facecolor = fig.patch.get_facecolor() # Use plot background for vector formats
                
                fig.savefig(filename, dpi=300, bbox_inches='tight',
                            facecolor=fig_facecolor,
                            transparent=filename.lower().endswith('.png')) # PNG can have transparency
                messagebox.showinfo("Export Successful", f"Chart saved as:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export chart:\n{str(e)}")

    def _toggle_radar_grid(self, ax, canvas): # Modified to accept canvas
        """Toggles the visibility of grid lines and labels on the radar chart."""
        if ax:
            # Toggle visibility of radial and angular grid lines
            for line in ax.yaxis.get_gridlines():
                line.set_visible(not line.get_visible())
            for line in ax.xaxis.get_gridlines():
                line.set_visible(not line.get_visible())

            # Toggle visibility of concentric circles (patches)
            for patch in ax.patches:
                if isinstance(patch, mpatches.Circle) and patch.get_radius() > 0:
                    patch.set_visible(not patch.get_visible())
            
            # Toggle visibility of y-axis tick labels
            for label in ax.get_yticklabels():
                label.set_visible(not label.get_visible())

            canvas.draw() # Redraw the canvas to apply changes

    def _show_fullscreen_chart(self, comp1_data, comp2_data, categories, comp1_values, comp2_values):
        """
        Opens a new Toplevel window to display the radar chart in fullscreen mode.
        """
        try:
            popup = tk.Toplevel(self.root)
            popup.title(f"Fullscreen Comparison: {comp1_data['name']} vs {comp2_data['name']}")
            popup.state('zoomed') # Maximize the window
            popup.configure(bg=COLORS['light'])
            
            # Create a new figure for the fullscreen display
            full_fig, full_ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
            full_fig.patch.set_facecolor(COLORS['background'])
            full_ax.set_facecolor('white')

            # Re-apply all radar chart elements and styling to the new figure
            grid_values = [2, 4, 6, 8, 10]
            full_ax.set_ylim(0, 10)
            full_ax.set_yticks(grid_values)
            full_ax.set_yticklabels([str(int(v)) for v in grid_values], fontsize=11, color=COLORS['text'], alpha=0.7)
            full_ax.grid(True, color=COLORS['grid'], alpha=0.7, linewidth=0.8, linestyle='--')
            
            for radius in grid_values:
                circle = mpatches.Circle((0, 0), radius, transform=full_ax.transData, fill=False, color=COLORS['grid'], alpha=0.5, linewidth=0.7, linestyle=':')
                full_ax.add_patch(circle)

            num_features = len(categories)
            angles = [n / float(num_features) * 2 * np.pi for n in range(num_features)]
            angles += angles[:1] # Close the loop

            # Plot data for both components
            full_ax.fill(angles, comp1_values + comp1_values[:1], color=COLORS['highlight'], alpha=0.3, label=comp1_data['name'])
            full_ax.plot(angles, comp1_values + comp1_values[:1], color=COLORS['highlight'], linewidth=3, marker='o', markersize=8, markerfacecolor='white', markeredgecolor=COLORS['highlight'], markeredgewidth=2)
            
            full_ax.fill(angles, comp2_values + comp2_values[:1], color=COLORS['success'], alpha=0.3, label=comp2_data['name'])
            full_ax.plot(angles, comp2_values + comp2_values[:1], color=COLORS['success'], linewidth=3, marker='s', markersize=8, markerfacecolor='white', markeredgecolor=COLORS['success'], markeredgewidth=2)
            
            # Set axis labels and title
            full_ax.set_xticks(angles[:-1])
            full_ax.set_xticklabels(categories, color=COLORS['text'], size=13, fontweight='bold')
            full_ax.set_title(f"Performance Comparison\n{comp1_data['name']} vs {comp2_data['name']}", color=COLORS['dark'], fontsize=18, fontweight='bold', va='bottom', pad=30)
            full_ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=14, frameon=True, fancybox=True, shadow=True, framealpha=0.95)

            # Add value labels
            for i, (angle, val1, val2) in enumerate(zip(angles[:-1], comp1_values, comp2_values)):
                full_ax.text(angle, val1 + 0.4, f'{val1:.1f}', ha='center', va='center', fontsize=9, color=COLORS['highlight'], fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS['highlight'], alpha=0.9, linewidth=0.5))
                
                va2 = 'bottom' if val2 < 1.5 else 'top'
                full_ax.text(angle, val2 - 0.4 if val2 > 1.5 else val2 + 0.4, f'{val2:.1f}', ha='center', va=va2, fontsize=9, color=COLORS['success'], fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS['success'], alpha=0.9, linewidth=0.5))

            # Add overall scores and winner text
            avg1 = sum(comp1_values) / len(comp1_values)
            avg2 = sum(comp2_values) / len(comp2_values)
            if avg1 > avg2: winner_text = f"🏆 {comp1_data['name']} leads"
            elif avg2 > avg1: winner_text = f"🏆 {comp2_data['name']} leads"
            else: winner_text = "🤝 Equal performance"
            summary_text = f"Overall Scores:\n{comp1_data['name']}: {avg1:.1f}/10\n{comp2_data['name']}: {avg2:.1f}/10\n\n{winner_text}"
            full_ax.text(0.02, 0.98, summary_text, transform=full_ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLORS['dark'], alpha=0.95, linewidth=2), color=COLORS['dark'], fontweight='bold')
            
            # Add strengths analysis text
            strengths_text = self._analyze_component_strengths(comp1_values, comp2_values, categories, comp1_data['name'], comp2_data['name'])
            full_ax.text(0.98, 0.02, strengths_text, transform=full_ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'], edgecolor=COLORS['dark'], alpha=0.9), color=COLORS['dark'])

            plt.tight_layout(pad=3.0) # Adjust layout for fullscreen

            # Embed the full figure into the popup window
            canvas = FigureCanvasTkAgg(full_fig, popup)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Close button for the fullscreen window
            close_btn = tk.Button(popup, text="Close Fullscreen", command=lambda: [popup.destroy(), plt.close(full_fig)], bg=COLORS['danger'], fg='white', font=('Arial', 12, 'bold'), relief=tk.FLAT)
            close_btn.pack(pady=10)
            
            canvas.draw()
            # Ensure figure is closed when window is closed via X button
            popup.protocol("WM_DELETE_WINDOW", lambda: [popup.destroy(), plt.close(full_fig)])
        except Exception as e:
            messagebox.showerror("Fullscreen Error", f"Failed to show fullscreen chart:\n{str(e)}")

    def open_comparison_module(self):
        """Switches the main notebook to the 'Component Comparison' tab."""
        self.main_content_notebook.select(self.compare_tab)

    def clear_comparison_display_content(self):
        """
        Clears all dynamic content from the component comparison display area.
        This includes images, table content, and the radar chart.
        Robustly checks if widgets are initialized and mapped before attempting to pack_forget.
        This function is now only called when the comparison window is to be explicitly closed,
        not when switching dropdowns on the main tab.
        """
        # This function is now mainly a cleanup function if needed,
        # but the primary cleanup is handled by _on_compare_window_close
        # when the Toplevel window is destroyed.
        pass


    def _clear_radar_chart_elements(self):
        """Internal helper to safely destroy Matplotlib radar chart elements."""
        # This function is now less critical as Toplevel destruction handles it,
        # but kept for consistency or if a specific plot needs to be cleared mid-process.
        # If it's called outside the Toplevel context, it won't have the canvas/toolbar references.
        # It's better to manage cleanup within the Toplevel's protocol.
        pass


if __name__ == "__main__":
    # Ensure the Tkinter root window is created and the app is initialized
    root = tk.Tk()
    app = PCBuilderApp(root)
    root.mainloop() # Start the Tkinter event loop

