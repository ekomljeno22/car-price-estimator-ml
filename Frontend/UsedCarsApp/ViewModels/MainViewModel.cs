using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using UsedCarsApp.Models;
using UsedCarsApp.Services;

namespace UsedCarsApp.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private readonly ICarApiService _api;

    // ── Dropdown option collections ───────────────────────────────────────────
    public ObservableCollection<string> FuelTypes     { get; } = [];
    public ObservableCollection<string> Transmissions { get; } = [];
    public ObservableCollection<string> Accidents     { get; } = [];
    public ObservableCollection<string> CleanTitles   { get; } = [];
    public ObservableCollection<string> Brands        { get; } = [];
    public ObservableCollection<string> CarModels     { get; } = [];

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
    [ObservableProperty] private string _predictedPrice    = "—";
    [ObservableProperty] private string _statusMessage     = "Loading options…";
    [ObservableProperty] private bool   _isBusy;
    [ObservableProperty] private bool   _hasError;

    public MainViewModel(ICarApiService api)
    {
        _api = api;
        _ = LoadOptionsAsync();
    }

    // ── Commands ──────────────────────────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanPredict))]
    private async Task PredictAsync()
    {
        IsBusy    = true;
        HasError  = false;
        StatusMessage = "Predicting…";

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
                StatusMessage  = "Prediction complete.";
            }
            else
            {
                PredictedPrice = "N/A";
                StatusMessage  = "No result returned.";
            }
        }
        catch (Exception ex)
        {
            HasError       = true;
            PredictedPrice = "Error";
            StatusMessage  = $"Error: {ex.Message}";
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

    // Keep CanExecute in sync with relevant property changes
    partial void OnSelectedBrandChanged(string value)    => PredictCommand.NotifyCanExecuteChanged();
    partial void OnSelectedCarModelChanged(string value) => PredictCommand.NotifyCanExecuteChanged();
    partial void OnIsBusyChanged(bool value)             => PredictCommand.NotifyCanExecuteChanged();

    // ── Options loader ────────────────────────────────────────────────────────

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
            Populate(CarModels,     options.Models);

            SelectedFuelType     = FuelTypes.FirstOrDefault("gas");
            SelectedTransmission = Transmissions.FirstOrDefault("automatic");
            SelectedAccident     = Accidents.FirstOrDefault("None");
            SelectedCleanTitle   = CleanTitles.FirstOrDefault("Yes");
            SelectedBrand        = Brands.FirstOrDefault(string.Empty) ?? string.Empty;
            SelectedCarModel     = CarModels.FirstOrDefault(string.Empty) ?? string.Empty;

            StatusMessage = "Ready. Fill in the form and click Predict.";
        }
        catch (Exception ex)
        {
            HasError      = true;
            StatusMessage = $"Could not load options: {ex.Message}";
        }
    }

    private static void Populate(ObservableCollection<string> target, IEnumerable<string> source)
    {
        target.Clear();
        foreach (var item in source) target.Add(item);
    }
}