"""Tiny keyword retriever over synthetic policy snippets.

Kept dependency-free on purpose; in production swap `retrieve()` for the hybrid retriever in
ops-agent (pgvector + FTS, RRF). The contract is the same: a list of {source, text}.
"""

from __future__ import annotations

import re

POLICIES: list[dict[str, str]] = [
    {
        "source": "sop_repasse.md",
        "text": "Repasse padrão em D+2 dias úteis para eventos; semanal (terça) para F&B. "
        "Chargebacks em disputa ficam retidos até 45 dias.",
    },
    {
        "source": "sop_reembolso.md",
        "text": "Reembolsos até R$ 500 podem ser aprovados pelo suporte N1 com aprovação humana; "
        "acima disso exigem o financeiro. Nunca reembolsar sem comprovante de cobrança duplicada.",
    },
    {
        "source": "sop_incidentes.md",
        "text": "Terminal sem sincronizar: verificar conectividade, forçar sincronização, abrir "
        "ticket alta após 30 min. Incidentes em portão durante evento são urgentes (15 min).",
    },
    {
        "source": "sop_acesso.md",
        "text": "Redefinição de senha e MFA são self-service no portal. Acesso de administrador "
        "exige aprovação do gestor da conta e fica registrado em auditoria.",
    },
    {
        "source": "politica_comercial.md",
        "text": "Descontos acima de 20% exigem diretor comercial. Renovação automática salvo "
        "aviso com 60 dias.",
    },
]

_TOKEN = re.compile(r"[\wÀ-ÿ]+")


def retrieve(query: str, k: int = 2) -> list[dict[str, str]]:
    q = {t.lower() for t in _TOKEN.findall(query) if len(t) > 3}
    scored = []
    for p in POLICIES:
        words = {t.lower() for t in _TOKEN.findall(p["text"])}
        # prefix match = cheap stemming (reembolso ~ reembolsos, cobrado ~ cobrança)
        overlap = sum(1 for qt in q if any(w.startswith(qt[:6]) for w in words))
        if overlap:
            scored.append((overlap, p))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in scored[:k]]
