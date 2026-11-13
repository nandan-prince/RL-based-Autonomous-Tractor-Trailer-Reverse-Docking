"""Shared dynamics contracts and continuous/discrete model utilities.

Concrete plant equations live in simulator-specific model modules.  This
module is reserved for controller-facing dynamics abstractions once NMPC is
implemented; it deliberately does not duplicate the kinematic plant.
"""
