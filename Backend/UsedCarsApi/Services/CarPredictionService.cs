using System.Net.Http.Json;
using System.Text.Json;
using UsedCarsApi.Models;

namespace UsedCarsApi.Services;

/// <summary>
/// Thin proxy that forwards requests to the Python FastAPI ML microservice.
/// </summary>
public interface ICarPredictionService
{
    Task<PredictResponse?> PredictAsync(PredictRequest request, CancellationToken ct = default);
    Task<CarOptions?>      GetOptionsAsync(CancellationToken ct = default);
}

public sealed class CarPredictionService(HttpClient http) : ICarPredictionService
{
    private static readonly JsonSerializerOptions _opts = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public async Task<PredictResponse?> PredictAsync(PredictRequest request, CancellationToken ct = default)
    {
        // Map C# PascalCase to Python snake_case
        var mlPayload = new
        {
            model_year   = request.ModelYear,
            milage       = request.Milage,
            fuel_type    = request.FuelType,
            transmission = request.Transmission,
            accident     = request.Accident,
            clean_title  = request.CleanTitle,
            brand        = request.Brand,
            model        = request.Model
        };

        var response = await http.PostAsJsonAsync("/predict", mlPayload, ct);
        response.EnsureSuccessStatusCode();

        var ml = await response.Content.ReadFromJsonAsync<MlPredictResponse>(_opts, ct);
        if (ml is null) return null;

        return new PredictResponse(ml.predicted_price, ml.predicted_price_formatted);
    }

    public async Task<CarOptions?> GetOptionsAsync(CancellationToken ct = default)
    {
        var ml = await http.GetFromJsonAsync<MlOptionsResponse>("/options", _opts, ct);
        if (ml is null) return null;

        return new CarOptions(
            ml.fuel_types,
            ml.transmissions,
            ml.accidents,
            ml.clean_titles,
            ml.brands,
            ml.models
        );
    }
}