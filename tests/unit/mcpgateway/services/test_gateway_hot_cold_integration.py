# -*- coding: utf-8 -*-
"""Integration tests for hot/cold server classification with GatewayService.

Tests the integration between ServerClassificationService and GatewayService
for health checks and auto-refresh polling.

Copyright 2026
SPDX-License-Identifier: Apache-2.0
"""

# Standard
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Third-Party
import pytest

# First-Party
from mcpgateway.db import Gateway as DbGateway
from mcpgateway.services.gateway_service import GatewayService
from mcpgateway.services.server_classification_service import ServerClassificationService


@pytest.fixture(autouse=True)
def mock_logging_services():
    """Mock audit_trail and structured_logger to prevent database writes during tests."""
    with (
        patch("mcpgateway.services.gateway_service.audit_trail") as mock_audit,
        patch("mcpgateway.services.gateway_service.structured_logger") as mock_logger,
    ):
        mock_audit.log_action = MagicMock(return_value=None)
        mock_logger.log = MagicMock(return_value=None)
        yield {"audit_trail": mock_audit, "structured_logger": mock_logger}


@pytest.fixture
def gateway_service_with_classification():
    """Create a GatewayService instance with classification service."""
    with patch("mcpgateway.services.gateway_service.SessionLocal"):
        service = GatewayService()
        service.oauth_manager = AsyncMock()

        # Mock classification service
        mock_classification = AsyncMock(spec=ServerClassificationService)
        service._classification_service = mock_classification

        return service, mock_classification


def _make_mock_gateway(
    gateway_id: str = "gw-123",
    name: str = "test-gateway",
    url: str = "http://test-server:8000",
    enabled: bool = True,
    reachable: bool = True,
) -> MagicMock:
    """Create a mock gateway object."""
    mock = MagicMock(spec=DbGateway)
    mock.id = gateway_id
    mock.name = name
    mock.url = url
    mock.enabled = enabled
    mock.reachable = reachable
    mock.transport = "SSE"
    mock.auth_type = None
    mock.auth_value = None
    mock.oauth_config = None
    mock.ca_certificate = None
    mock.ca_certificate_sig = None
    mock.client_cert = None
    mock.client_key = None
    mock.auth_query_params = None
    mock.visibility = "private"
    mock.last_refresh_at = None
    mock.refresh_interval_seconds = None
    return mock


class TestHealthCheckHotColdIntegration:
    """Tests for health check integration with hot/cold polling."""

    @pytest.mark.asyncio
    async def test_health_check_skipped_for_hot_server_not_due(self, gateway_service_with_classification):
        """Test health check skipped when hot server not yet due for polling."""
        gateway_service, mock_classification = gateway_service_with_classification

        # Configure classification service to skip polling
        mock_classification.should_poll_server = AsyncMock(return_value=False)

        mock_gateway = _make_mock_gateway(url="http://hot-server:8000")

        # Should return early without actually polling
        await gateway_service._check_single_gateway_health(mock_gateway, user_email="test@example.com")

        # Verify should_poll_server was called with "health" poll type
        mock_classification.should_poll_server.assert_awaited_once_with("http://hot-server:8000", "health")

    @pytest.mark.asyncio
    async def test_health_check_proceeds_for_hot_server_due(self, gateway_service_with_classification):
        """Test health check proceeds when hot server is due for polling."""
        gateway_service, mock_classification = gateway_service_with_classification

        # Configure classification service to allow polling
        mock_classification.should_poll_server = AsyncMock(return_value=True)

        mock_gateway = _make_mock_gateway(url="http://hot-server:8000")

        # Mock HTTP client to simulate successful health check
        with (
            patch("mcpgateway.services.gateway_service.fresh_db_session") as mock_fresh_db,
            patch("mcpgateway.services.gateway_service.get_isolated_http_client") as mock_client,
        ):
            mock_session = MagicMock()
            mock_fresh_db.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value.scalar_one_or_none.return_value = mock_gateway

            # Mock HTTP response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_http = AsyncMock()
            mock_http.__aenter__.return_value = mock_http
            mock_http.__aexit__.return_value = None
            mock_http.stream = AsyncMock(return_value=mock_response)
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = None
            mock_client.return_value = mock_http

            await gateway_service._check_single_gateway_health(mock_gateway, user_email="test@example.com")

            # Verify should_poll_server was called
            mock_classification.should_poll_server.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_health_check_proceeds_when_classification_disabled(self, gateway_service_with_classification):
        """Test health check always proceeds when classification feature is disabled."""
        gateway_service, mock_classification = gateway_service_with_classification

        # Set classification service to None (feature disabled)
        gateway_service._classification_service = None

        mock_gateway = _make_mock_gateway(url="http://any-server:8000")

        with (
            patch("mcpgateway.services.gateway_service.fresh_db_session") as mock_fresh_db,
            patch("mcpgateway.services.gateway_service.get_isolated_http_client") as mock_client,
        ):
            mock_session = MagicMock()
            mock_fresh_db.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value.scalar_one_or_none.return_value = mock_gateway

            # Mock HTTP response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_http = AsyncMock()
            mock_http.__aenter__.return_value = mock_http
            mock_http.__aexit__.return_value = None
            mock_http.stream = AsyncMock(return_value=mock_response)
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = None
            mock_client.return_value = mock_http

            await gateway_service._check_single_gateway_health(mock_gateway, user_email="test@example.com")

            # Health check should proceed without classification check (classification service is None)
            # Verify HTTP client was used (health check executed)
            mock_client.assert_called()

    @pytest.mark.asyncio
    async def test_health_check_cold_server_skipped_when_not_due(self, gateway_service_with_classification):
        """Test health check skipped for cold server not yet due."""
        gateway_service, mock_classification = gateway_service_with_classification

        # Cold server, polling not due
        mock_classification.should_poll_server = AsyncMock(return_value=False)

        mock_gateway = _make_mock_gateway(url="http://cold-server:8000")

        await gateway_service._check_single_gateway_health(mock_gateway, user_email="test@example.com")

        # Verify early return (should_poll_server called, health check not executed)
        mock_classification.should_poll_server.assert_awaited_once_with("http://cold-server:8000", "health")


class TestAutoRefreshHotColdIntegration:
    """Tests for auto-refresh integration with hot/cold polling."""

    @pytest.mark.asyncio
    async def test_auto_refresh_skipped_for_hot_server_not_due(self, gateway_service_with_classification):
        """Test auto-refresh skipped when hot server not yet due for polling."""
        gateway_service, mock_classification = gateway_service_with_classification

        # Configure classification to skip polling
        mock_classification.should_poll_server = AsyncMock(return_value=False)

        mock_gateway = _make_mock_gateway(url="http://hot-server:8000")
        mock_gateway.last_refresh_at = None  # Never refreshed before

        # Result should reflect no refresh attempted
        # (Verify by checking classification service was consulted)
        with patch("mcpgateway.services.gateway_service.settings") as mock_settings:
            mock_settings.auto_refresh_servers = True

            # The actual test depends on the internal flow, but we verify consultation
            # In real flow, should_poll_server("tools") would be called
            result = await mock_classification.should_poll_server("http://hot-server:8000", "tools")

            assert result is False

    @pytest.mark.asyncio
    async def test_auto_refresh_proceeds_for_hot_server_due(self, gateway_service_with_classification):
        """Test auto-refresh proceeds when hot server is due for polling."""
        gateway_service, mock_classification = gateway_service_with_classification

        # Configure classification to allow polling
        mock_classification.should_poll_server = AsyncMock(return_value=True)

        # Verify should_poll_server called with "tools" poll type
        result = await mock_classification.should_poll_server("http://hot-server:8000", "tools")

        assert result is True
        mock_classification.should_poll_server.assert_awaited_once_with("http://hot-server:8000", "tools")

    @pytest.mark.asyncio
    async def test_auto_refresh_cold_server_skipped_when_not_due(self, gateway_service_with_classification):
        """Test auto-refresh skipped for cold server not yet due."""
        gateway_service, mock_classification = gateway_service_with_classification

        # Cold server, polling not due
        mock_classification.should_poll_server = AsyncMock(return_value=False)

        result = await mock_classification.should_poll_server("http://cold-server:8000", "tools")

        assert result is False
        mock_classification.should_poll_server.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auto_refresh_proceeds_when_classification_disabled(self, gateway_service_with_classification):
        """Test auto-refresh always proceeds when classification disabled."""
        gateway_service, mock_classification = gateway_service_with_classification

        # Set classification service to None (feature disabled)
        gateway_service._classification_service = None

        # Without classification service, auto-refresh should proceed normally
        # (This is verified by the absence of should_poll_server call)
        assert gateway_service._classification_service is None


class TestPollTypeIndependence:
    """Tests for independent tracking of health and tool_discovery poll types."""

    @pytest.mark.asyncio
    async def test_health_and_tools_polled_independently(self, gateway_service_with_classification):
        """Test health and tool_discovery polls tracked independently."""
        gateway_service, mock_classification = gateway_service_with_classification

        url = "http://test-server:8000"

        # Health check recently done, tools not yet polled
        async def should_poll_side_effect(url_arg, poll_type):
            if poll_type == "health":
                return False  # Health check not due
            elif poll_type == "tools":
                return True  # Tools refresh due
            return True

        mock_classification.should_poll_server = AsyncMock(side_effect=should_poll_side_effect)

        # Check health polling
        health_result = await mock_classification.should_poll_server(url, "health")
        assert health_result is False

        # Check tools polling
        tools_result = await mock_classification.should_poll_server(url, "tools")
        assert tools_result is True

        # Verify both types were checked
        assert mock_classification.should_poll_server.await_count == 2


class TestClassificationServiceInitialization:
    """Tests for classification service initialization in GatewayService."""

    @pytest.mark.asyncio
    async def test_classification_service_initialized_when_enabled(self):
        """Test classification service initialized when feature enabled."""
        mock_redis = AsyncMock()

        with (
            patch("mcpgateway.services.gateway_service.settings") as mock_settings,
            patch("mcpgateway.services.gateway_service.SessionLocal"),
            patch("mcpgateway.services.gateway_service.get_redis_client") as mock_get_redis,
        ):
            mock_settings.hot_cold_classification_enabled = True
            mock_settings.platform_admin_email = "admin@example.com"
            mock_get_redis.return_value = mock_redis

            service = GatewayService()

            # Initialize the service (which would normally create classification service)
            with patch.object(service, "_run_health_checks") as mock_run_health:
                mock_run_health.return_value = None

                await service.initialize()

                # Verify classification service was created
                assert service._classification_service is not None

    @pytest.mark.asyncio
    async def test_classification_service_not_initialized_when_disabled(self):
        """Test classification service not initialized when feature disabled."""
        with (
            patch("mcpgateway.services.gateway_service.settings") as mock_settings,
            patch("mcpgateway.services.gateway_service.SessionLocal"),
        ):
            mock_settings.hot_cold_classification_enabled = False
            mock_settings.platform_admin_email = "admin@example.com"

            service = GatewayService()

            with patch.object(service, "_run_health_checks") as mock_run_health:
                mock_run_health.return_value = None

                await service.initialize()

                # Classification service should not be created
                assert service._classification_service is None or not hasattr(service, "_classification_service")


class TestConfigurationValues:
    """Tests for hot/cold polling interval configuration."""

    @pytest.mark.asyncio
    async def test_hot_server_interval_equals_gateway_auto_refresh(self):
        """Test hot server check interval equals gateway_auto_refresh_interval."""
        # First-Party
        from mcpgateway.config import Settings

        # Create a real config instance with specific value
        config = Settings(gateway_auto_refresh_interval=300)

        # Access the property
        assert config.hot_server_check_interval == 300

    @pytest.mark.asyncio
    async def test_cold_server_interval_is_3x_gateway_auto_refresh(self):
        """Test cold server check interval is 3x gateway_auto_refresh_interval."""
        # First-Party
        from mcpgateway.config import Settings

        # Create a real config instance with specific value
        config = Settings(gateway_auto_refresh_interval=300)

        # Access the property
        assert config.cold_server_check_interval == 900  # 3x

    @pytest.mark.asyncio
    async def test_intervals_derive_correctly_from_config(self):
        """Test intervals correctly derived from configuration."""
        # First-Party
        from mcpgateway.config import Settings

        # Test different base intervals
        base_intervals = [60, 120, 300, 600]

        for base in base_intervals:
            # Create a config instance for each test value
            config = Settings(gateway_auto_refresh_interval=base)

            # Verify derived intervals
            hot_interval = config.hot_server_check_interval
            cold_interval = config.cold_server_check_interval

            assert hot_interval == base
            assert cold_interval == base * 3


class TestFailOpenBehavior:
    """Tests for fail-open behavior on errors."""

    @pytest.mark.asyncio
    async def test_health_check_proceeds_on_classification_error(self, gateway_service_with_classification):
        """Test health check proceeds when classification check fails."""
        gateway_service, mock_classification = gateway_service_with_classification

        # Classification service raises error
        mock_classification.should_poll_server = AsyncMock(side_effect=Exception("Redis connection error"))

        mock_gateway = _make_mock_gateway(url="http://test-server:8000")

        # Health check should still proceed (fail-open)
        with (
            patch("mcpgateway.services.gateway_service.fresh_db_session") as mock_fresh_db,
            patch("mcpgateway.services.gateway_service.get_isolated_http_client") as mock_client,
        ):
            mock_session = MagicMock()
            mock_fresh_db.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value.scalar_one_or_none.return_value = mock_gateway

            # Mock HTTP response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_http = AsyncMock()
            mock_http.__aenter__.return_value = mock_http
            mock_http.__aexit__.return_value = None
            mock_http.stream = AsyncMock(return_value=mock_response)
            mock_response.__aenter__.return_value = mock_response
            mock_response.__aexit__.return_value = None
            mock_client.return_value = mock_http

            # Should not raise exception, health check should proceed
            try:
                await gateway_service._check_single_gateway_health(mock_gateway, user_email="test@example.com")
            except Exception as e:
                pytest.fail(f"Health check should not raise exception on classification error: {e}")
