using System.Net.Http.Json;
using System.Text.Json;
using UsedCarsApi.Models;

namespace UsedCarsApi.Services;

public interface ICarPredictionService
{
    Task<PredictResponse?>      PredictAsync(PredictRequest request, CancellationToken ct = default);
    Task<CarOptions?>           GetOptionsAsync(CancellationToken ct = default);
    Task<BrandModelsResponse?>  GetModelsForBrandAsync(string brand, CancellationToken ct = default);
    Task<ModelStats?>           GetStatsAsync(CancellationToken ct = default);

    Task<ChartsResponse?>  GetChartsAsync(CancellationToken ct = default);
    Task<byte[]?>          GetChartImageAsync(string filename, CancellationToken ct = default);
}

public sealed class CarPredictionService(HttpClient http) : ICarPredictionService
{
    private static readonly JsonSerializerOptions _opts = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public async Task<PredictResponse?> PredictAsync(PredictRequest request, CancellationToken ct = default)
    {
        var payload = new
        {
            model_year   = request.ModelYear,
            milage       = request.Milage,
            fuel_type    = request.FuelType,
            transmission = request.Transmission,
            accident     = request.Accident,
            clean_title  = request.CleanTitle,
            brand        = request.Brand,
            model        = request.Model,
            hp           = request.Hp,
            liters       = request.Liters
        };

        var response = await http.PostAsJsonAsync("/predict", payload, ct);
        response.EnsureSuccessStatusCode();

        var ml = await response.Content.ReadFromJsonAsync<MlPredictResponse>(_opts, ct);
        return ml is null ? null : new PredictResponse(
            ml.predicted_price,
            ml.predicted_price_formatted,
            ml.model_used
        );
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

    public async Task<BrandModelsResponse?> GetModelsForBrandAsync(string brand, CancellationToken ct = default)
    {
        var encoded = Uri.EscapeDataString(brand);
        var ml      = await http.GetFromJsonAsync<MlBrandModelsResponse>(
                          $"/models-for-brand/{encoded}", _opts, ct);
        return ml is null ? null : new BrandModelsResponse(ml.brand, ml.models);
    }

    public async Task<ModelStats?> GetStatsAsync(CancellationToken ct = default)
    {
        var ml = await http.GetFromJsonAsync<MlStatsResponse>("/stats", _opts, ct);
        return ml is null ? null : new ModelStats(
            ml.model_type,
            ml.mae,
            ml.rmse,
            ml.r2,
            ml.mape,
            ml.train_mape,
            ml.train_r2,
            ml.training_samples,
            ml.best_model
        );
    }

    public async Task<ChartsResponse?> GetChartsAsync(CancellationToken ct = default)
    {
        var ml = await http.GetFromJsonAsync<MlChartsResponse>("/charts", _opts, ct);
        if (ml is null) return null;

        var dtos = ml.charts
            .Select(c => new ChartDto(c.filename, c.label, c.url))
            .ToList();

        return new ChartsResponse(dtos);
    }

    public async Task<byte[]?> GetChartImageAsync(string filename, CancellationToken ct = default)
    {
        var encoded = Uri.EscapeDataString(filename);
        return await http.GetByteArrayAsync($"/charts/{encoded}", ct);
    }
}