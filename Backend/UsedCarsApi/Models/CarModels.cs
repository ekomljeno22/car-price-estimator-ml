namespace UsedCarsApi.Models;

public record PredictRequest
(
    int    ModelYear,
    float  Milage,
    float  Hp,
    float  Liters,
    string FuelType,
    string Transmission,
    string Accident,
    string CleanTitle,
    string Brand,
    string Model
);

public record PredictResponse(
    double PredictedPrice,
    string PredictedPriceFormatted,
    string ModelUsed
);

public record CarOptions(
    IReadOnlyList<string> FuelTypes,
    IReadOnlyList<string> Transmissions,
    IReadOnlyList<string> Accidents,
    IReadOnlyList<string> CleanTitles,
    IReadOnlyList<string> Brands,
    IReadOnlyList<string> Models
);

public record BrandModelsResponse(
    string                Brand,
    IReadOnlyList<string> Models
);

public record ModelStats(
    string  ModelType,
    double? Mae,
    double? Rmse,
    double? R2,
    double? Mape,
    double? TrainMape,
    double? TrainR2,
    int?    TrainingSamples,
    string? BestModel
);

public record ChartDto(
    string Filename,
    string Label,
    string Url
);

public record ChartsResponse(
    IReadOnlyList<ChartDto> Charts
);

internal record MlPredictResponse(
    double predicted_price,
    string predicted_price_formatted,
    string model_used
);

internal record MlOptionsResponse(
    IReadOnlyList<string> fuel_types,
    IReadOnlyList<string> transmissions,
    IReadOnlyList<string> accidents,
    IReadOnlyList<string> clean_titles,
    IReadOnlyList<string> brands,
    IReadOnlyList<string> models
);

internal record MlBrandModelsResponse(
    string                brand,
    IReadOnlyList<string> models
);

internal record MlStatsResponse(
    string  model_type,
    double? mae,
    double? rmse,
    double? r2,
    double? mape,
    double? train_mape,
    double? train_r2,
    int?    training_samples,
    string? best_model
);

internal record MlChartItem(
    string filename,
    string label,
    string url
);

internal record MlChartsResponse(
    IReadOnlyList<MlChartItem> charts
);