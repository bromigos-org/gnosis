"""HTTP route registration, grouped by API surface.

Each module exposes a ``register_*_routes`` function that
:func:`gnosis.main.create_app` calls with the app, settings, authenticator,
and backend accessor. The legacy ``/v1/context`` routes stay in
``gnosis.main`` because their deprecation warning is logged under that
module's logger.
"""
