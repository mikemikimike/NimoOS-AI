package v2

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/NimoTech/NimoOS-AI/service"
	"github.com/labstack/echo/v4"
	"github.com/stretchr/testify/require"
)

type modelsTestServices struct {
	service.Services
	checker *service.OllamaChecker
	manager *service.ModelManager
}

func (s *modelsTestServices) OllamaChecker() *service.OllamaChecker { return s.checker }
func (s *modelsTestServices) ModelManager() *service.ModelManager   { return s.manager }

func TestModelsHandler_List_MissingUserID_Returns401(t *testing.T) {
	e := echo.New()
	h := &ModelsHandler{}

	req := httptest.NewRequest(http.MethodGet, "/v1/ai/models", nil)
	rec := httptest.NewRecorder()
	c := e.NewContext(req, rec)

	err := h.List(c)
	var httpErr *echo.HTTPError
	require.ErrorAs(t, err, &httpErr)
	require.Equal(t, http.StatusUnauthorized, httpErr.Code)
}

func TestModelsHandler_Delete_MissingUserID_Returns401(t *testing.T) {
	e := echo.New()
	h := &ModelsHandler{}

	req := httptest.NewRequest(http.MethodDelete, "/v1/ai/models/llama3:8b", nil)
	rec := httptest.NewRecorder()
	c := e.NewContext(req, rec)

	err := h.Delete(c)
	var httpErr *echo.HTTPError
	require.ErrorAs(t, err, &httpErr)
	require.Equal(t, http.StatusUnauthorized, httpErr.Code)
}

func TestModelsHandler_Pull_OllamaUnavailable_Returns503(t *testing.T) {
	var pulls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/pull" {
			pulls.Add(1)
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	svc := &modelsTestServices{
		checker: service.NewOllamaChecker(server.URL),
		manager: service.NewModelManager(server.URL, nil, nil),
	}
	h := NewModelsHandler(svc, t.TempDir())
	e := echo.New()
	req := httptest.NewRequest(http.MethodPost, "/v1/ai/models/pull", strings.NewReader(`{"name":"qwen3:0.6b"}`))
	req.Header.Set("X-NimoOS-User-ID", "1")
	req.Header.Set(echo.HeaderContentType, echo.MIMEApplicationJSON)
	rec := httptest.NewRecorder()
	c := e.NewContext(req, rec)

	err := h.Pull(c)
	var httpErr *echo.HTTPError
	require.ErrorAs(t, err, &httpErr)
	require.Equal(t, http.StatusServiceUnavailable, httpErr.Code)
	require.Equal(t, "Ollama is not running", httpErr.Message)
	require.Zero(t, pulls.Load())
}

// Regression: Ollama HF tags like `hf.co/unsloth/X:UD-IQ2_M` arrive in the URL
// as `hf.co%2Funsloth%2FX%3AUD-IQ2_M`. Echo's c.Param("name") returns the raw
// percent-encoded form — the Delete handler MUST url.PathUnescape it before
// forwarding to Ollama, otherwise Ollama rejects the name as invalid.
func TestModelsHandler_Delete_PathParamPreservesEncoding(t *testing.T) {
	e := echo.New()
	var captured string
	e.DELETE("/v1/ai/models/:name", func(c echo.Context) error {
		captured = c.Param("name")
		return c.NoContent(http.StatusNoContent)
	})
	encoded := "hf.co%2Funsloth%2Fgemma-4-E4B-it-GGUF%3AUD-IQ2_M"
	req := httptest.NewRequest(http.MethodDelete, "/v1/ai/models/"+encoded, nil)
	rec := httptest.NewRecorder()
	e.ServeHTTP(rec, req)

	require.Equal(t, encoded, captured)
	decoded, err := url.PathUnescape(captured)
	require.NoError(t, err)
	require.Equal(t, "hf.co/unsloth/gemma-4-E4B-it-GGUF:UD-IQ2_M", decoded)
}
