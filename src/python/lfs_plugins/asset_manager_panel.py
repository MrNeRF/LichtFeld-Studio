# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager panel for browsing and managing Gaussian Splatting assets."""

import lichtfeld as lf
from .types import Panel

__lfs_panel_classes__ = ["AssetManagerPanel"]
__lfs_panel_ids__ = ["lfs.asset_manager"]


class AssetManagerPanel(Panel):
    """Floating Asset Manager window for browsing splats, videos, and exports."""

    id = "lfs.asset_manager"
    label = "Asset Manager"
    space = lf.ui.PanelSpace.FLOATING
    order = 20
    template = "rmlui/asset_manager.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    size = (980, 620)
    update_interval_ms = 500

    # Mock asset data for interactive testing
    MOCK_ASSETS = [
        {
            "id": "dante_statue_01",
            "name": "dante_statue_01",
            "type": "ply",
            "size": "182 MB",
            "points": "1,000,000",
            "tags": ["statue", "dante", "outdoor"],
            "thumb_class": "asset-thumb-statue",
            "thumb_label": "Statue",
            "is_favorite": True,
        },
        {
            "id": "fountain_square_02",
            "name": "fountain_square_02",
            "type": "ply",
            "size": "412 MB",
            "points": "2,450,000",
            "tags": ["architecture", "outdoor"],
            "thumb_class": "asset-thumb-fountain",
            "thumb_label": "Fountain",
            "is_favorite": False,
        },
        {
            "id": "museum_hall_01",
            "name": "museum_hall_01",
            "type": "ply",
            "size": "563 MB",
            "points": "3,120,000",
            "tags": ["interior", "museum"],
            "thumb_class": "asset-thumb-interior",
            "thumb_label": "Interior",
            "is_favorite": False,
        },
        {
            "id": "olive_tree_01",
            "name": "olive_tree_01",
            "type": "ply",
            "size": "128 MB",
            "points": "850,000",
            "tags": ["nature", "outdoor"],
            "thumb_class": "asset-thumb-tree",
            "thumb_label": "Tree",
            "is_favorite": False,
        },
        {
            "id": "piazza_flythrough",
            "name": "piazza_flythrough",
            "type": "mp4",
            "size": "98 MB",
            "resolution": "1920x1080",
            "tags": ["video", "drone"],
            "thumb_class": "asset-thumb-video",
            "thumb_label": "Video",
            "duration": "00:15",
            "is_favorite": False,
        },
        {
            "id": "bust_scanned_01",
            "name": "bust_scanned_01",
            "type": "rad",
            "size": "540 MB",
            "tags": ["scan", "bust", "interior"],
            "thumb_class": "asset-thumb-bust",
            "thumb_label": "Bust",
            "is_favorite": True,
        },
    ]

    # Filter definitions with counts
    FILTERS = [
        {"id": "all", "label": "All Assets", "count": 24},
        {"id": "splat", "label": "Splats", "count": 14},
        {"id": "video", "label": "Videos", "count": 6},
        {"id": "recent", "label": "Recent", "count": 10},
        {"id": "favorites", "label": "Favorites", "count": 5},
    ]

    # Tag definitions
    TAGS = [
        {"id": "architecture", "label": "architecture", "count": 8},
        {"id": "statue", "label": "statue", "count": 6},
        {"id": "outdoor", "label": "outdoor", "count": 7},
        {"id": "interior", "label": "interior", "count": 4},
        {"id": "scan", "label": "scan", "count": 12},
        {"id": "dante", "label": "dante", "count": 2},
    ]

    # Collection definitions
    COLLECTIONS = [
        {"id": "city_scans", "label": "City Scans", "count": 6},
        {"id": "statues", "label": "Statues", "count": 5},
        {"id": "interiors", "label": "Interiors", "count": 4},
        {"id": "tests", "label": "Tests", "count": 3},
        {"id": "reference", "label": "Reference", "count": 2},
    ]

    def __init__(self):
        self._handle = None
        self._doc = None
        # State tracking
        self._selected_ids = set()  # Set of selected asset IDs
        self._active_filter = "all"  # Current filter ID
        self._active_tab = "info"  # Current info tab: info, parameters, history
        self._view_mode = "gallery"  # Current view mode: gallery, list
        self._search_query = ""  # Current search query

    # ── Data model ────────────────────────────────────────────

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("asset_manager")
        if model is None:
            return

        # Basic properties
        model.bind_func("panel_label", lambda: "Asset Manager")
        model.bind_func("search_query", self.get_search_query)
        model.bind_func("asset_count", self.get_asset_count)
        model.bind_func("collection_count", lambda: 8)
        model.bind_func("selected_count", self.get_selected_count)
        model.bind_func("selected_total_size", self.get_selected_total_size)

        # View state
        model.bind_func("view_mode", self.get_view_mode)
        model.bind_func("is_gallery_view", lambda: self._view_mode == "gallery")
        model.bind_func("is_list_view", lambda: self._view_mode == "list")

        # Active states
        model.bind_func("active_filter", self.get_active_filter)
        model.bind_func("active_tab", self.get_active_tab)

        # Lists (use bind_record_list for data-for loops)
        model.bind_record_list("filters")
        model.bind_record_list("tags")
        model.bind_record_list("collections")
        model.bind_record_list("assets")

        self._handle = model.get_handle()

        # Initialize record lists immediately so data-for loops have data
        self._update_record_lists()

        # Formatted strings for display
        model.bind_func(
            "selected_count_text", lambda: f"{len(self._selected_ids)} selected"
        )
        model.bind_func(
            "selected_total_text", lambda: f"Total: {self.get_selected_total_size()}"
        )

        # Event handlers
        model.bind_event("set_filter", self.set_filter)
        model.bind_event("set_tab", self.set_tab)
        model.bind_event("set_view_mode", self.set_view_mode)
        model.bind_event("toggle_asset_selection", self.toggle_asset_selection)
        model.bind_event("on_search", self.on_search)

    # ── Data getters ─────────────────────────────────────────

    def get_search_query(self):
        return self._search_query

    def get_asset_count(self):
        return len(self.get_filtered_assets())

    def get_selected_count(self):
        return len(self._selected_ids)

    def get_selected_total_size(self):
        # Mock size calculation
        total_mb = 0
        for asset in self.MOCK_ASSETS:
            if asset["id"] in self._selected_ids:
                size_str = asset.get("size", "0 MB")
                try:
                    mb = int(size_str.split()[0])
                    total_mb += mb
                except (ValueError, IndexError):
                    pass
        if total_mb >= 1024:
            return f"{total_mb / 1024:.2f} GB"
        return f"{total_mb} MB"

    def get_view_mode(self):
        return self._view_mode

    def get_active_filter(self):
        return self._active_filter

    def get_active_tab(self):
        return self._active_tab

    def get_filters(self):
        return self.FILTERS

    def get_tags(self):
        return self.TAGS

    def get_collections(self):
        return self.COLLECTIONS

    def get_filtered_assets(self):
        """Return assets filtered by search query and active filter."""
        assets = self.MOCK_ASSETS.copy()

        # Apply search filter
        if self._search_query:
            query = self._search_query.lower()
            assets = [
                a
                for a in assets
                if query in a["name"].lower()
                or any(query in tag.lower() for tag in a.get("tags", []))
            ]

        # Apply category filter
        if self._active_filter == "splat":
            assets = [a for a in assets if a["type"] in ("ply", "rad")]
        elif self._active_filter == "video":
            assets = [a for a in assets if a["type"] == "mp4"]
        elif self._active_filter == "recent":
            # Mock: first 4 assets are "recent"
            recent_ids = {a["id"] for a in self.MOCK_ASSETS[:4]}
            assets = [a for a in assets if a["id"] in recent_ids]
        elif self._active_filter == "favorites":
            # Mock: dante_statue_01 and bust_scanned_01 are favorites
            fav_ids = {"dante_statue_01", "bust_scanned_01"}
            assets = [a for a in assets if a["id"] in fav_ids]

        # Add selection state to each asset
        for asset in assets:
            asset["is_selected"] = asset["id"] in self._selected_ids

        return assets

    # ── Event handlers ───────────────────────────────────────

    def set_filter(self, _handle, _ev, args):
        """Set the active filter."""
        if not args:
            return
        filter_id = str(args[0])
        self._active_filter = filter_id
        self._dirty_model("active_filter", "filters", "assets", "asset_count")

    def set_tab(self, _handle, _ev, args):
        """Set the active info tab."""
        if not args:
            return
        tab_id = str(args[0])
        self._active_tab = tab_id
        self._dirty_model("active_tab")

    def set_view_mode(self, _handle, _ev, args):
        """Set the view mode (gallery or list)."""
        if not args:
            return
        mode = str(args[0])
        self._view_mode = mode
        self._dirty_model("view_mode", "is_gallery_view", "is_list_view")

    def toggle_asset_selection(self, _handle, _ev, args):
        """Toggle selection state of an asset."""
        if not args:
            return
        asset_id = str(args[0])
        if asset_id in self._selected_ids:
            self._selected_ids.remove(asset_id)
        else:
            self._selected_ids.add(asset_id)
        self._dirty_model("assets", "selected_count", "selected_total_size")

    def on_search(self, _handle, _ev, args):
        """Handle search input change."""
        # Search query is updated via two-way binding, just refresh assets
        self._dirty_model("search_query", "assets", "asset_count")

    # ── Lifecycle ─────────────────────────────────────────────

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        # Initialize record lists for data-for loops
        self._update_record_lists()

    def on_update(self, doc):
        # Data model handles updates via dirty marking
        return False

    def on_unmount(self, doc):
        doc.remove_data_model("asset_manager")
        self._handle = None
        self._doc = None

    # ── Helpers ──────────────────────────────────────────────

    def _update_record_lists(self):
        """Update all record lists in the data model."""
        if not self._handle:
            return
        # Update arrays used by data-for loops
        self._handle.update_record_list("filters", self.FILTERS)
        self._handle.update_record_list("tags", self.TAGS)
        self._handle.update_record_list("collections", self.COLLECTIONS)
        self._handle.update_record_list("assets", self.get_filtered_assets())

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            self._update_record_lists()
            return
        for field in fields:
            self._handle.dirty(field)
            # Update record lists when they change
            if field in ("filters", "tags", "collections", "assets"):
                self._update_record_lists()
