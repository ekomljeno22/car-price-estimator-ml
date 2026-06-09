using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using UsedCarsApp.Models;

namespace UsedCarsApp.Services;

public interface ICarApiService
{
    Task<PredictResponse?>      PredictAsync(PredictRequest request, CancellationToken ct = default);
    Task<CarOptions?>           GetOptionsAsync(CancellationToken ct = default);
    Task<IReadOnlyList<string>> GetModelsForBrandAsync(string brand, CancellationToken ct = default);
    Task<ModelStats?>           GetStatsAsync(CancellationToken ct = default);
    Task<List<ChartDto>?>       GetChartsAsync(CancellationToken ct = default);
    Task<byte[]?>               GetChartImageAsync(string relativeUrl, CancellationToken ct = default);
}

public sealed class CarApiService : ICarApiService
{
    private readonly HttpClient _http;

    private static readonly JsonSerializerOptions _opts = new()
    {
        PropertyNamingPolicy        = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition      = JsonIgnoreCondition.WhenWritingNull
    };

    public CarApiService(string baseUrl = "http://localhost:5000")
    {
        _http = new HttpClient { BaseAddress = new Uri(baseUrl) };
    }

    public async Task<PredictResponse?> PredictAsync(PredictRequest request, CancellationToken ct = default)
    {
        var response = await _http.PostAsJsonAsync("/api/cars/predict", request, _opts, ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<PredictResponse>(_opts, ct);
    }

    public async Task<CarOptions?> GetOptionsAsync(CancellationToken ct = default)
        => await _http.GetFromJsonAsync<CarOptions>("/api/cars/options", _opts, ct);

    public async Task<IReadOnlyList<string>> GetModelsForBrandAsync(string brand, CancellationToken ct = default)
    {
        var encoded  = Uri.EscapeDataString(brand);
        var response = await _http.GetFromJsonAsync<BrandModelsResponse>(
                           $"/api/cars/models-for-brand/{encoded}", _opts, ct);
        return response?.Models ?? [];
    }

    public async Task<ModelStats?> GetStatsAsync(CancellationToken ct = default)
        => await _http.GetFromJsonAsync<ModelStats>("/api/cars/stats", _opts, ct);

    public async Task<List<ChartDto>?> GetChartsAsync(CancellationToken ct = default)
    {
        var response = await _http.GetFromJsonAsync<ChartsResponse>("/api/cars/charts", _opts, ct);
        return response?.Charts;
    }

    public async Task<byte[]?> GetChartImageAsync(string relativeUrl, CancellationToken ct = default)
        => await _http.GetByteArrayAsync(relativeUrl, ct);

    private sealed record ChartsResponse(
        [property: JsonPropertyName("charts")] List<ChartDto> Charts
    );
}