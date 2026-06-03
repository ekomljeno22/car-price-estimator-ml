using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using UsedCarsApp.Models;

namespace UsedCarsApp.Services;

public interface ICarApiService
{
    Task<PredictResponse?> PredictAsync(PredictRequest request, CancellationToken ct = default);
    Task<CarOptions?>      GetOptionsAsync(CancellationToken ct = default);
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
}