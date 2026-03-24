# -*- coding: utf-8 -*-
"""
Server Classification Service.

Manages hot/cold server classification based on MCP session pool usage patterns.
Provides staggered polling to optimize resource allocation and reduce polling overhead.

Classification is based ONLY on upstream MCP pooled session state (gateway -> MCP servers).

Copyright 2026
SPDX-License-Identifier: Apache-2.0
"""

# flake8: noqa: DAR101, DAR201, DAR401

# Future
from __future__ import annotations

# Standard
import asyncio
from dataclasses import asdict, dataclass
import hashlib
import logging
from math import floor
import time
from typing import Dict, List, Literal, Optional, TYPE_CHECKING

# Third-Party
import orjson

# First-Party
from mcpgateway.config import settings

if TYPE_CHECKING:
    # Third-Party
    from redis.asyncio import Redis

    # First-Party
    from mcpgateway.services.mcp_session_pool import MCPSessionPool

logger = logging.getLogger(__name__)


@dataclass
class ServerUsageMetrics:
    """Aggregated usage metrics for a single server from pooled sessions."""

    url: str
    server_last_used: float = 0.0  # max(last_used) across all pooled sessions
    active_session_count: int = 0  # Count from _active dict
    total_use_count: int = 0  # Sum of use_count from all sessions
    pooled_session_count: int = 0  # Total pooled sessions for this server


@dataclass
class ClassificationMetadata:
    """Metadata about classification run."""

    total_servers: int  # Total servers
    hot_cap: int  # Maximum hot servers (20% of total_servers)
    hot_actual: int  # Actual hot servers selected
    eligible_count: int  # Servers with pooled sessions
    timestamp: float  # Classification timestamp
    underutilized_reason: Optional[str] = None  # Why hot < 20% (if applicable)


@dataclass
class ClassificationResult:
    """Result of server classification."""

    hot_servers: List[str]  # URLs of hot servers
    cold_servers: List[str]  # URLs of cold servers
    metadata: ClassificationMetadata


class ServerClassificationService:
    """
    Manages hot/cold server classification based on MCP session pool state.

    Classification Logic:
        1. Scope: Uses only upstream MCP pooled session state
        2. Hot cap: floor(20% * total_servers)
        3. Eligibility: Server must have pooled session with valid last_used
        4. Ranking: server_last_used descending (newest first)
        5. Tie-breakers: active_count, use_count, URL (deterministic)
        6. Hot selection: Top min(hot_cap, eligible_count)
        7. Cold: All remaining servers
        8. Guarantees: No overlap, full coverage, deterministic

    Thread-safe for multi-worker deployments via Redis state management.
    Falls back to local-only operation when Redis unavailable.
    """

    # Redis key templates
    CLASSIFICATION_HOT_KEY = "mcpgateway:server_classification:hot"
    CLASSIFICATION_COLD_KEY = "mcpgateway:server_classification:cold"
    CLASSIFICATION_METADATA_KEY = "mcpgateway:server_classification:metadata"
    CLASSIFICATION_TIMESTAMP_KEY = "mcpgateway:server_classification:timestamp"
    POLL_STATE_KEY_TEMPLATE = "mcpgateway:server_poll_state:{url_hash}:last_{poll_type}"
    LEADER_KEY = "mcpgateway:server_classification:leader"

    def __init__(self, redis_client: Optional[Redis] = None):
        """Initialize classification service.

        Args:
            redis_client: Redis client for state management (optional for single-worker)
        """
        self._redis = redis_client
        self._classification_task: Optional[asyncio.Task] = None
        self._instance_id = f"classifier_{id(self)}"
        self._leader_ttl = 90  # seconds
        self._running = False
        self._error_backoff_seconds: float = 30.0  # Back off duration on loop errors (override in tests)

    async def start(self) -> None:
        """Start background classification loop (if enabled)."""
        if not settings.hot_cold_classification_enabled:
            logger.info("Hot/cold classification disabled")
            return

        if self._running:
            logger.warning("Classification service already running")
            return

        self._running = True
        self._classification_task = asyncio.create_task(self._run_classification_loop())
        logger.info(f"Server classification service started " f"(instance={self._instance_id}, redis={'enabled' if self._redis else 'disabled'})")

    async def stop(self) -> None:
        """Stop background classification."""
        self._running = False
        if self._classification_task:
            self._classification_task.cancel()
            try:
                await self._classification_task
            except asyncio.CancelledError:
                logger.info("Classification task cancelled")

    async def _run_classification_loop(self) -> None:
        """Background loop: classify servers periodically with leader election."""
        while self._running:
            try:
                # Leader election (Redis-based for multi-worker, local-only otherwise)
                is_leader = await self._try_acquire_leader_lock()

                if is_leader:
                    logger.debug(f"Classification leader acquired (instance={self._instance_id})")
                    await self._perform_classification()
                else:
                    logger.debug(f"Not classification leader, skipping (instance={self._instance_id})")

                await asyncio.sleep(settings.gateway_auto_refresh_interval)

            except asyncio.CancelledError:
                logger.info("Classification loop cancelled")
                break
            except Exception as e:
                logger.error(f"Classification loop error: {e}", exc_info=True)
                await asyncio.sleep(self._error_backoff_seconds)  # Back off on error

    async def _try_acquire_leader_lock(self) -> bool:
        """Try to acquire leader lock for classification.

        Returns:
            True if this instance is leader, False otherwise
        """
        if not self._redis:
            # Single-worker mode (no Redis), always leader
            return True

        try:
            # Try to acquire leader key (expires after TTL)
            is_leader = await self._redis.set(self.LEADER_KEY, self._instance_id, ex=self._leader_ttl, nx=True)  # Only if key doesn't exist
            return bool(is_leader)
        except Exception as e:
            logger.warning(f"Failed to acquire leader lock: {e}")
            return False  # Fail safe: don't classify on error

    async def _perform_classification(self) -> None:
        """Perform classification and publish to Redis (if available)."""
        try:
            # Get MCP session pool
            # First-Party
            from mcpgateway.services.mcp_session_pool import get_mcp_session_pool

            try:
                pool = get_mcp_session_pool()
            except RuntimeError:
                logger.debug("MCP session pool not initialized, skipping classification")
                return

            # Get all gateway URLs from database
            all_gateway_urls = await self._get_all_gateway_urls()
            if not all_gateway_urls:
                logger.debug("No gateways found, skipping classification")
                return

            # Perform classification
            result = self._classify_servers_from_pool(pool, all_gateway_urls)

            # Publish to Redis (if available)
            if self._redis:
                await self._publish_classification_to_redis(result)

            logger.info(
                f"Classification completed: {len(result.hot_servers)} hot, " f"{len(result.cold_servers)} cold (N={result.metadata.total_servers}, " f"eligible={result.metadata.eligible_count})"
            )

            if result.metadata.underutilized_reason:
                logger.debug(f"Underutilization: {result.metadata.underutilized_reason}")

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)

    def _classify_servers_from_pool(self, pool: MCPSessionPool, all_gateway_urls: List[str]) -> ClassificationResult:
        """Classify servers based on pooled session state.

        Algorithm (deterministic):
            1. Get total servers N
            2. Calculate hot_cap = floor(0.20 * N)
            3. Extract server metrics from pooled sessions
            4. Filter eligible (has valid last_used)
            5. Sort by (server_last_used desc, active_count desc, use_count desc, url asc)
            6. Select top min(hot_cap, eligible_count) as hot
            7. Remaining servers are cold

        Args:
            pool: MCP session pool
            all_gateway_urls: All registered gateway URLs

        Returns:
            ClassificationResult with hot/cold servers and metadata
        """
        total_servers = len(all_gateway_urls)
        hot_cap = floor(0.20 * total_servers)

        # Step 3: Extract server usage from pooled sessions
        server_metrics: Dict[str, ServerUsageMetrics] = {}

        # Iterate over pool._pools (Dict[PoolKey, Queue[PooledSession]])
        # PoolKey = (user_identity, url, identity_hash, transport_type, gateway_id)
        for pool_key, session_queue in pool._pools.items():  # pylint: disable=protected-access
            url = pool_key[1]  # Extract server URL from pool key

            if url not in server_metrics:
                server_metrics[url] = ServerUsageMetrics(url=url)

            # Process each pooled session in the queue
            try:
                # Access queue items (asyncio.Queue has internal _queue deque)
                sessions_list = list(session_queue._queue) if hasattr(session_queue, "_queue") else []  # pylint: disable=protected-access

                for session in sessions_list:
                    # PooledSession has: last_used, use_count
                    if hasattr(session, "last_used") and session.last_used > 0:
                        # Update server-level last_used to most recent
                        server_metrics[url].server_last_used = max(server_metrics[url].server_last_used, session.last_used)
                        server_metrics[url].total_use_count += getattr(session, "use_count", 0)
                        server_metrics[url].pooled_session_count += 1
            except Exception as e:
                logger.warning(f"Error extracting metrics for {url}: {e}")
                continue

        # Count active sessions from _active dict
        for pool_key, active_set in pool._active.items():  # pylint: disable=protected-access
            url = pool_key[1]
            if url in server_metrics:
                server_metrics[url].active_session_count += len(active_set)

        # Step 4: Filter eligible servers (has valid last_used)
        eligible_servers = [metrics for metrics in server_metrics.values() if metrics.server_last_used > 0.0]
        eligible_count = len(eligible_servers)

        # Step 5: Sort by recency (newer first), then tie-breakers
        eligible_servers.sort(
            key=lambda m: (
                -m.server_last_used,  # Primary: most recent first (descending)
                -m.active_session_count,  # Tie-breaker 1: more active sessions
                -m.total_use_count,  # Tie-breaker 2: higher use count
                m.url,  # Tie-breaker 3: deterministic (ascending)
            )
        )

        # Step 6: Select hot servers (up to hot_cap, no backfill)
        hot_actual = min(hot_cap, eligible_count)
        hot_servers = [m.url for m in eligible_servers[:hot_actual]]

        # Step 7: Cold servers = all remaining
        hot_set = set(hot_servers)
        cold_servers = [url for url in all_gateway_urls if url not in hot_set]

        # Step 8: Build metadata
        underutilized_reason = None
        if eligible_count < hot_cap:
            underutilized_reason = f"Only {eligible_count} servers have pooled sessions, " f"below hot_cap={hot_cap}"

        return ClassificationResult(
            hot_servers=hot_servers,
            cold_servers=cold_servers,
            metadata=ClassificationMetadata(
                total_servers=total_servers, hot_cap=hot_cap, hot_actual=hot_actual, eligible_count=eligible_count, timestamp=time.time(), underutilized_reason=underutilized_reason
            ),
        )

    async def _get_all_gateway_urls(self) -> List[str]:
        """Get all enabled gateway URLs from database.

        Returns:
            List of gateway URLs
        """
        # Third-Party
        from sqlalchemy import select

        # First-Party
        from mcpgateway.db import Gateway, SessionLocal

        try:
            with SessionLocal() as db:
                result = db.execute(select(Gateway.url).where(Gateway.enabled.is_(True)))
                urls = [row[0] for row in result]
                return urls
        except Exception as e:
            logger.error(f"Failed to get gateway URLs: {e}")
            return []

    async def _publish_classification_to_redis(self, result: ClassificationResult) -> None:
        """Publish classification result to Redis atomically.

        Args:
            result: Classification result to publish
        """
        if not self._redis:
            return

        try:
            # Atomic pipeline for transactional updates
            async with self._redis.pipeline(transaction=True) as pipe:
                # Clear old classification
                await pipe.delete(self.CLASSIFICATION_HOT_KEY, self.CLASSIFICATION_COLD_KEY)

                # Set new classification
                # Set TTL on classification sets to prevent stale data after worker crash
                ttl = int(settings.gateway_auto_refresh_interval * 2)

                if result.hot_servers:
                    await pipe.sadd(self.CLASSIFICATION_HOT_KEY, *result.hot_servers)

                if result.cold_servers:
                    await pipe.sadd(self.CLASSIFICATION_COLD_KEY, *result.cold_servers)

                # Expire classification sets regardless of whether they had members
                await pipe.expire(self.CLASSIFICATION_HOT_KEY, ttl)
                await pipe.expire(self.CLASSIFICATION_COLD_KEY, ttl)

                # Store metadata (expire after 2x classification interval)
                metadata_json = orjson.dumps(asdict(result.metadata))
                await pipe.set(self.CLASSIFICATION_METADATA_KEY, metadata_json, ex=ttl)

                await pipe.set(self.CLASSIFICATION_TIMESTAMP_KEY, result.metadata.timestamp, ex=ttl)

                await pipe.execute()

            logger.debug("Classification published to Redis successfully")

        except Exception as e:
            logger.error(f"Failed to publish classification to Redis: {e}")

    async def get_server_classification(self, url: str) -> Optional[str]:
        """Get classification for a server (hot/cold).

        Args:
            url: Server URL

        Returns:
            "hot", "cold", or None if not classified
        """
        if not self._redis:
            return None  # No Redis, classification not available

        try:
            is_hot = await self._redis.sismember(self.CLASSIFICATION_HOT_KEY, url)
            if is_hot:
                return "hot"

            is_cold = await self._redis.sismember(self.CLASSIFICATION_COLD_KEY, url)
            if is_cold:
                return "cold"

            return None  # Not yet classified
        except Exception as e:
            logger.warning(f"Failed to get classification for {url}: {e}")
            return None  # Fail open

    async def should_poll_server(self, url: str, poll_type: Literal["health", "tool_discovery"]) -> bool:
        """Determine if server should be polled now based on classification.

        Args:
            url: Server URL
            poll_type: Type of poll (health or tool_discovery)

        Returns:
            True if should poll now, False otherwise
        """
        if not settings.hot_cold_classification_enabled:
            return True  # Feature disabled, always poll

        if not self._redis:
            return True  # No Redis, always poll (single-worker mode)

        try:
            classification = await self.get_server_classification(url)
            if classification is None:
                return True  # Not yet classified, poll anyway

            # Get last poll time (hash URL to prevent key injection and reduce key size)
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:32]
            last_poll_key = self.POLL_STATE_KEY_TEMPLATE.format(url_hash=url_hash, poll_type=poll_type)
            last_poll_str = await self._redis.get(last_poll_key)

            if last_poll_str is None:
                # Never polled, should poll now
                await self._update_poll_timestamp(url, poll_type, classification)
                return True

            last_poll = float(last_poll_str)
            now = time.time()
            if not (0 < last_poll <= now + 60):
                last_poll = 0.0  # treat as never polled; prevents manipulation via future timestamps
            elapsed = now - last_poll

            # Determine interval based on classification
            interval = settings.hot_server_check_interval if classification == "hot" else settings.cold_server_check_interval

            should_poll = elapsed >= interval

            if should_poll:
                await self._update_poll_timestamp(url, poll_type, classification)

            return should_poll

        except Exception as e:
            logger.warning(f"Error checking poll status for {url}: {e}")
            return True  # Fail open: poll on error

    async def _update_poll_timestamp(self, url: str, poll_type: str, classification: str) -> None:
        """Update last poll timestamp in Redis.

        Args:
            url: Server URL
            poll_type: Type of poll
            classification: Server classification (hot/cold)
        """
        if not self._redis:
            return

        interval = settings.hot_server_check_interval if classification == "hot" else settings.cold_server_check_interval

        try:
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:32]
            last_poll_key = self.POLL_STATE_KEY_TEMPLATE.format(url_hash=url_hash, poll_type=poll_type)
            await self._redis.set(last_poll_key, time.time(), ex=int(interval * 2))  # Expire after 2x interval
        except Exception as e:
            logger.warning(f"Failed to update poll timestamp for {url}: {e}")
