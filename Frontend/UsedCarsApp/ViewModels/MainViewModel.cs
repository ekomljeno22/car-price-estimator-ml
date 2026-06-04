using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using UsedCarsApp.Models;
using UsedCarsApp.Services;

namespace UsedCarsApp.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private readonly ICarApiService _api;

    // ── Dropdown kolekcije ────────────────────────────────────────────────────
    public ObservableCollection<string> FuelTypes     { get; } = [];
    public ObservableCollection<string> Transmissions { get; } = [];
    public ObservableCollection<string> Accidents     { get; } = [];
    public ObservableCollection<string> CleanTitles   { get; } = [];
    public ObservableCollection<string> Brands        { get; } = [];

    // CarModels je ODVOJEN od ostalih – puni se dinamički po odabranom brandu
    public ObservableCollection<string> CarModels { get; } = [];

    // ── Input fields ──────────────────────────────────────────────────────────
    [ObservableProperty] private int    _modelYear    = DateTime.Now.Year - 3;
    [ObservableProperty] private double _milage       = 50_000;
    [ObservableProperty] private string _selectedFuelType     = "gas";
    [ObservableProperty] private string _selectedTransmission = "automatic";
    [ObservableProperty] private string _selectedAccident     = "None";
    [ObservableProperty] private string _selectedCleanTitle   = "Yes";
    [ObservableProperty] private string _selectedBrand        = string.Empty;
    [ObservableProperty] private string _selectedCarModel     = string.Empty;

    // ── Output / status ───────────────────────────────────────────────────────
    [ObservableProperty] private string _predictedPrice = "—";
    [ObservableProperty] private string _statusMessage  = "Učitavanje opcija…";
    [ObservableProperty] private bool   _isBusy;
    [ObservableProperty] private bool   _hasError;
    [ObservableProperty] private bool   _isLoadingModels;

    public MainViewModel(ICarApiService api)
    {
        _api = api;
        _ = LoadOptionsAsync();
    }

    // ── Kad se brand promijeni → učitaj filtrirane modele ────────────────────
    partial void OnSelectedBrandChanged(string value)
    {
        PredictCommand.NotifyCanExecuteChanged();

        // Prazni modele odmah da korisnik ne može slučajno odabrati krivi model
        CarModels.Clear();
        SelectedCarModel = string.Empty;

        if (!string.IsNullOrEmpty(value))
            _ = LoadModelsForBrandAsync(value);
    }

    partial void OnSelectedCarModelChanged(string value) =>
        PredictCommand.NotifyCanExecuteChanged();

    partial void OnIsBusyChanged(bool value) =>
        PredictCommand.NotifyCanExecuteChanged();

    // ── Commands ──────────────────────────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanPredict))]
    private async Task PredictAsync()
    {
        IsBusy    = true;
        HasError  = false;
        StatusMessage = "Predviđam cijenu…";

        try
        {
            var request = new PredictRequest(
                ModelYear:    ModelYear,
                Milage:       Milage,
                FuelType:     SelectedFuelType,
                Transmission: SelectedTransmission,
                Accident:     SelectedAccident,
                CleanTitle:   SelectedCleanTitle,
                Brand:        SelectedBrand,
                Model:        SelectedCarModel
            );

            var result = await _api.PredictAsync(request);
            if (result is not null)
            {
                PredictedPrice = result.PredictedPriceFormatted;
                StatusMessage  = "Predviđanje završeno.";
            }
            else
            {
                PredictedPrice = "N/A";
                StatusMessage  = "Servis nije vratio rezultat.";
            }
        }
        catch (Exception ex)
        {
            HasError       = true;
            PredictedPrice = "Greška";
            StatusMessage  = $"Greška: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    private bool CanPredict() =>
        !IsBusy &&
        !string.IsNullOrEmpty(SelectedBrand) &&
        !string.IsNullOrEmpty(SelectedCarModel);

    // ── Učitaj inicijalne opcije (brandovi, fuel, transmission…) ─────────────
    private async Task LoadOptionsAsync()
    {
        try
        {
            var options = await _api.GetOptionsAsync();
            if (options is null) return;

            Populate(FuelTypes,     options.FuelTypes);
            Populate(Transmissions, options.Transmissions);
            Populate(Accidents,     options.Accidents);
            Populate(CleanTitles,   options.CleanTitles);
            Populate(Brands,        options.Brands);

            SelectedFuelType     = FuelTypes.FirstOrDefault("gas");
            SelectedTransmission = Transmissions.FirstOrDefault("automatic");
            SelectedAccident     = Accidents.FirstOrDefault("None");
            SelectedCleanTitle   = CleanTitles.FirstOrDefault("Yes");

            // Postavljanje prvog branda triggerat će OnSelectedBrandChanged
            // koji će automatski učitati modele za taj brand
            if (Brands.Count > 0)
                SelectedBrand = Brands[0];

            StatusMessage = "Odaberi brand i model automobila.";
        }
        catch (Exception ex)
        {
            HasError      = true;
            StatusMessage = $"Greška pri učitavanju opcija: {ex.Message}";
        }
    }

    // ── Učitaj modele za odabrani brand ───────────────────────────────────────
    private async Task LoadModelsForBrandAsync(string brand)
    {
        IsLoadingModels = true;
        StatusMessage   = $"Učitavam modele za {brand}…";

        try
        {
            var models = await _api.GetModelsForBrandAsync(brand);

            CarModels.Clear();
            foreach (var m in models)
                CarModels.Add(m);

            // Automatski odaberi prvi model
            SelectedCarModel = CarModels.Count > 0 ? CarModels[0] : string.Empty;

            StatusMessage = CarModels.Count > 0
                ? $"{CarModels.Count} modela za {brand}. Klikni Predvidi."
                : $"Nema modela za brand '{brand}'.";
        }
        catch (Exception ex)
        {
            HasError      = true;
            StatusMessage = $"Greška pri učitavanju modela: {ex.Message}";
        }
        finally
        {
            IsLoadingModels = false;
            PredictCommand.NotifyCanExecuteChanged();
        }
    }

    private static void Populate(ObservableCollection<string> target, IEnumerable<string> source)
    {
        target.Clear();
        foreach (var item in source) target.Add(item);
    }
}