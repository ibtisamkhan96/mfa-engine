# Runs mfa-engine's Streamlit dashboard as a standalone container. Local dev checks
# out dk-wind-mfa and crm-trade-network as sibling folders next to this repo (see
# app.py's own DK_WIND_MFA_SRC/CRM_TRADE_NETWORK_SRC comment); a container has no such
# thing, so this clones both real, public sibling repos at build time instead of
# vendoring a copy.
#
# Each clone is pinned to an exact commit. Two reasons. Reproducibility: the image is built
# from the same sibling versions the tests passed against, not whatever is newest. And caching:
# an unpinned clone line never changes, so a builder with layer caching (Railway) reuses an old
# clone and silently misses upstream data; this happened once, when lithium moved to a new HS
# code and the deployed app kept an old crm-trade-network without it. Bumping a commit below
# changes this layer and forces a fresh clone.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

ARG DK_WIND_MFA_COMMIT=2ae16dee66b3fca2e13edb5cc96ee71b29ccd500
ARG CRM_TRADE_NETWORK_COMMIT=a84ee78de13306bc150a54aeaa038e3e8e887b44
RUN git clone --quiet https://github.com/ibtisamkhan96/dk-wind-mfa.git /deps/dk-wind-mfa \
 && git -C /deps/dk-wind-mfa checkout --quiet "$DK_WIND_MFA_COMMIT" \
 && git clone --quiet https://github.com/ibtisamkhan96/crm-trade-network.git /deps/crm-trade-network \
 && git -C /deps/crm-trade-network checkout --quiet "$CRM_TRADE_NETWORK_COMMIT"

COPY requirements.txt .
# pandas pinned below 3.0, not left to float: dk-wind-mfa's own load.py assigns pd.NA
# into a bool-dtype column (its own real, working pattern under pandas 2.x, the major
# version every test in this session has actually run against). pandas 3.0 tightened
# that assignment to raise TypeError instead, a real cross-version break found by
# building this image (an unpinned `pandas>=2.0` resolved to the newest 3.x release),
# not a hypothetical one. Pinning here matches the pandas major version this code is
# actually verified against, rather than silently taking on an untested one.
RUN pip install --no-cache-dir "pandas<3.0" \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r /deps/dk-wind-mfa/requirements.txt \
 && pip install --no-cache-dir -r /deps/crm-trade-network/requirements.txt

COPY . .

ENV DK_WIND_MFA_SRC=/deps/dk-wind-mfa/src
ENV CRM_TRADE_NETWORK_SRC=/deps/crm-trade-network/src

EXPOSE 8501
# Shell form (not exec form) so ${PORT:-8501} actually expands: Railway/Render inject a
# real $PORT at runtime, and this falls back to 8501 for a plain `docker run` locally.
CMD streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true
