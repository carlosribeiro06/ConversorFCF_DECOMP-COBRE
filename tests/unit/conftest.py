"""Fixtures shared by the unit suite."""

from collections.abc import Callable

import pytest

from conversor_fcf.decomp.mapcut_writer import MapcutHeader


@pytest.fixture
def mapcut_header() -> Callable[..., MapcutHeader]:
    """Build a small internally consistent header: 3 stages, 5 nodes, 4 plants.

    `max_lag` is deliberately absent from the defaults, so every test that uses
    this factory exercises its documented default of 0.
    """

    def build(**overrides: object) -> MapcutHeader:
        defaults: dict[str, object] = {
            "numero_iteracoes": 4,
            "numero_cortes": 12,
            "numero_submercados": 5,
            "numero_uhes": 4,
            "numero_cenarios": 5,
            "numero_estagios": 3,
            "numero_semanas": 3,
            "n_utv": 0,
            "dia": 25,
            "mes": 4,
            "ano": 2026,
            "codigos_uhes": (1, 2, 4, 6),
            "codigos_uhes_jusante": (2, 4, 6, 0),
            "indice_no_arvore": (1, 1, 2, 3, 3),
            "indice_primeiro_no_estagio": (1, 2, 3),
            "patamares_por_estagio": (3, 3, 3),
            "registro_ultimo_corte_no": (12, 11, 10, 0, 0),
            "codigos_submercados_gnl": (1, 3),
            "lag_meses_gnl": (2, 2),
            "patamares_gnl": (3, 3),
            "taxa_desconto": (1.0, 0.9978, 0.9957),
        }
        defaults.update(overrides)
        return MapcutHeader(**defaults)  # type: ignore[arg-type]

    return build
