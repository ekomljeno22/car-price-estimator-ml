namespace UsedCarsApp.Models;

public record PredictRequest
(
    int ModelYear,
    double Milage,
    float Hp,
    float Liters,
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
    string ModelType,
    double? Mae,
    double? Rmse,
    double? R2,
    double? Mape,
    double? TrainMape,
    double? TrainR2,
    int?    TrainingSamples,
    string? BestModel
);