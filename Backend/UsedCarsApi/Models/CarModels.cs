namespace UsedCarsApi.Models;

public record PredictRequest(
    int    ModelYear,
    double Milage,
    string FuelType,
    string Transmission,
    string Accident,
    string CleanTitle,
    string Brand,
    string Model
);

public record PredictResponse(
    double PredictedPrice,
    string PredictedPriceFormatted
);

public record CarOptions(
    IReadOnlyList<string> FuelTypes,
    IReadOnlyList<string> Transmissions,
    IReadOnlyList<string> Accidents,
    IReadOnlyList<string> CleanTitles,
    IReadOnlyList<string> Brands,
    IReadOnlyList<string> Models
);

// NOVO: odgovor za modele unutar branda
public record BrandModelsResponse(
    string               Brand,
    IReadOnlyList<string> Models
);

// ML service response shapes
internal record MlPredictResponse(
    double predicted_price,
    string predicted_price_formatted
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
    string               brand,
    IReadOnlyList<string> models
);