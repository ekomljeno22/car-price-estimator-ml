namespace UsedCarsApp.Models;

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

// NOVO: odgovor za modele filtriranje po brandu
public record BrandModelsResponse(
    string                Brand,
    IReadOnlyList<string> Models
);