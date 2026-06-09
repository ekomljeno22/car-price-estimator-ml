using System.Collections.ObjectModel;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Styling;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using UsedCarsApp.Models;
using UsedCarsApp.Services;

namespace UsedCarsApp.ViewModels;

public partial class ChartItem : ObservableObject
{
    public string Label { get; init; } = string.Empty;
    public string Url   { get; init; } = string.Empty;

    [ObservableProperty] private Bitmap? _imageSource;
}

public partial class MainViewModel : ObservableObject
{
    private readonly ICarApiService _api;

    public ObservableCollection<string>    FuelTypes     { get; } = [];
    public ObservableCollection<string>    Transmissions { get; } = [];
    public ObservableCollection<string>    Accidents     { get; } = [];
    public ObservableCollection<string>    CleanTitles   { get; } = [];
    public ObservableCollection<string>    Brands        { get; } = [];
    public ObservableCollection<string>    CarModels     { get; } = [];
    public ObservableCollection<StatCard>  StatsCards    { get; } = [];
    public ObservableCollection<ChartItem> ChartItems    { get; } = [];
    public ObservableCollection<ThemeOption> ThemeOptions { get; } = [];

    [ObservableProperty] private int    _modelYear            = DateTime.Now.Year - 3;
    [ObservableProperty] private double _milage               = 50_000;
    [ObservableProperty] private string _selectedFuelType     = "gas";
    [ObservableProperty] private string _selectedTransmission = "automatic";
    [ObservableProperty] private string _selectedAccident     = "None";
    [ObservableProperty] private string _selectedCleanTitle   = "Yes";
    [ObservableProperty] private string _selectedBrand        = string.Empty;
    [ObservableProperty] private string _selectedCarModel     = string.Empty;
    [ObservableProperty] private double _hp                   = 200;
    [ObservableProperty] private double _liters               = 2.0;

    [ObservableProperty] private string _predictedPrice = "—";
    [ObservableProperty] private string _modelUsed      = "—";
    [ObservableProperty] private string _statsModelType = "—";
    [ObservableProperty] private string _statusMessage  = "Učitavanje opcija…";
    [ObservableProperty] private bool   _isBusy;
    [ObservableProperty] private bool   _hasError;
    [ObservableProperty] private bool   _isLoadingModels;
    [ObservableProperty] private bool   _showResultPopup;
    [ObservableProperty] private bool   _hasCharts;
    [ObservableProperty] private bool   _isLoadingCharts;
    [ObservableProperty] private bool   _isChartViewerOpen;
    [ObservableProperty] private ChartItem? _selectedChart;
    [ObservableProperty] private double _chartZoom = 1.0;
    [ObservableProperty] private double _chartImageWidth;
    [ObservableProperty] private double _chartImageHeight;
    [ObservableProperty] private ThemeOption? _selectedTheme;

    public MainViewModel(ICarApiService api)
    {
        _api = api;
        LoadThemes();
        _ = LoadOptionsAsync();
        _ = LoadStatsAsync();
    }

    partial void OnSelectedThemeChanged(ThemeOption? value) => ApplyTheme(value);

    partial void OnSelectedChartChanged(ChartItem? value) => UpdateChartImageSize();

    partial void OnChartZoomChanged(double value) => UpdateChartImageSize();

    partial void OnSelectedBrandChanged(string value)
    {
        PredictCommand.NotifyCanExecuteChanged();
        CarModels.Clear();
        SelectedCarModel = string.Empty;
        if (!string.IsNullOrEmpty(value))
            _ = LoadModelsForBrandAsync(value);
    }

    partial void OnSelectedCarModelChanged(string value) =>
        PredictCommand.NotifyCanExecuteChanged();

    partial void OnIsBusyChanged(bool value) =>
        PredictCommand.NotifyCanExecuteChanged();

    [RelayCommand]
    private void ClosePopup() => ShowResultPopup = false;

    [RelayCommand]
    private void OpenChart(ChartItem? chart)
    {
        if (chart?.ImageSource is null) return;

        SelectedChart = chart;
        ChartZoom = 1.0;
        IsChartViewerOpen = true;
    }

    [RelayCommand]
    private void CloseChartViewer() => IsChartViewerOpen = false;

    [RelayCommand]
    private void PreviousChart()
    {
        if (ChartItems.Count == 0) return;

        var index = SelectedChart is null ? 0 : ChartItems.IndexOf(SelectedChart);
        SelectedChart = ChartItems[(index <= 0 ? ChartItems.Count : index) - 1];
        ChartZoom = 1.0;
    }

    [RelayCommand]
    private void NextChart()
    {
        if (ChartItems.Count == 0) return;

        var index = SelectedChart is null ? -1 : ChartItems.IndexOf(SelectedChart);
        SelectedChart = ChartItems[(index + 1) % ChartItems.Count];
        ChartZoom = 1.0;
    }

    [RelayCommand]
    private void ZoomInChart() => ChartZoom = Math.Min(3.0, ChartZoom + 0.25);

    [RelayCommand]
    private void ZoomOutChart() => ChartZoom = Math.Max(0.5, ChartZoom - 0.25);

    [RelayCommand]
    private void ResetChartZoom() => ChartZoom = 1.0;

    private void UpdateChartImageSize()
    {
        if (SelectedChart?.ImageSource is null)
        {
            ChartImageWidth = 0;
            ChartImageHeight = 0;
            return;
        }

        ChartImageWidth = SelectedChart.ImageSource.PixelSize.Width * ChartZoom;
        ChartImageHeight = SelectedChart.ImageSource.PixelSize.Height * ChartZoom;
    }

    [RelayCommand(CanExecute = nameof(CanPredict))]
    private async Task PredictAsync()
    {
        IsBusy        = true;
        HasError      = false;
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
                Model:        SelectedCarModel,
                Hp:           (float)Hp,
                Liters:       (float)Liters
            );

            var result = await _api.PredictAsync(request);
            if (result is not null)
            {
                PredictedPrice  = result.PredictedPriceFormatted;
                ModelUsed       = string.IsNullOrWhiteSpace(result.ModelUsed) ? "—" : result.ModelUsed;
                StatusMessage   = "Predviđanje završeno.";
                ShowResultPopup = true;
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

    [RelayCommand]
    private async Task LoadChartsAsync()
    {
        if (IsLoadingCharts) return;

        IsLoadingCharts = true;
        HasCharts       = false;
        ChartItems.Clear();
        StatusMessage = "Dohvaćam grafove s ML servisa…";

        try
        {
            var chartDtos = await _api.GetChartsAsync();
            if (chartDtos is null || chartDtos.Count == 0)
            {
                StatusMessage = "Nema grafova na servisu.";
                return;
            }

            foreach (var dto in chartDtos)
            {
                var item = new ChartItem { Label = dto.Label, Url = dto.Url };
                ChartItems.Add(item);
            }

            HasCharts     = ChartItems.Count > 0;
            StatusMessage = $"Učitano {ChartItems.Count} grafova.";

            await Task.WhenAll(ChartItems.Select(async item =>
            {
                try
                {
                    var imageBytes = await _api.GetChartImageAsync(item.Url);
                    if (imageBytes is { Length: > 0 })
                    {
                        await using var ms = new MemoryStream(imageBytes);
                        item.ImageSource = new Bitmap(ms);
                    }
                }
                catch
                {
                    // Slika se neće prikazati, ali ostale nastavljaju
                }
            }));
        }
        catch (Exception ex)
        {
            HasError      = true;
            StatusMessage = $"Greška pri dohvatu grafova: {ex.Message}";
        }
        finally
        {
            IsLoadingCharts = false;
        }
    }

    private void LoadThemes()
    {
        ThemeOptions.Add(new ThemeOption(
            "Dark Inferno",
            ThemeVariant.Dark,
            "#060B12", "#0D1829", "#0A1520", "#1E3048",
            "#F8FAFC", "#CBD5E1", "#64748B",
            "#F97316", "#FB923C", "#38BDF8", "#B0060B12"));

        ThemeOptions.Add(new ThemeOption(
            "Dark Ocean",
            ThemeVariant.Dark,
            "#051316", "#0B2027", "#071A20", "#16404A",
            "#ECFEFF", "#BAE6FD", "#6B8B96",
            "#14B8A6", "#2DD4BF", "#F59E0B", "#B0051316"));

        ThemeOptions.Add(new ThemeOption(
            "Light Sprint",
            ThemeVariant.Light,
            "#F8FAFC", "#FFFFFF", "#EEF2F7", "#CBD5E1",
            "#0F172A", "#334155", "#64748B",
            "#EA580C", "#F97316", "#0284C7", "#B8F8FAFC"));

        ThemeOptions.Add(new ThemeOption(
            "Light Mint",
            ThemeVariant.Light,
            "#F5FBF7", "#FFFFFF", "#EAF7EF", "#B7DEC6",
            "#102018", "#315743", "#6B806F",
            "#059669", "#10B981", "#7C3AED", "#B8F5FBF7"));

        SelectedTheme = ThemeOptions[0];
    }

    private static void ApplyTheme(ThemeOption? theme)
    {
        if (theme is null || Application.Current is null) return;

        Application.Current.RequestedThemeVariant = theme.Variant;
        SetBrush("AppBackgroundBrush", theme.Background);
        SetBrush("CardBrush", theme.Card);
        SetBrush("CardInnerBrush", theme.CardInner);
        SetBrush("BorderSoftBrush", theme.Border);
        SetBrush("TextMainBrush", theme.Text);
        SetBrush("TextSoftBrush", theme.TextSoft);
        SetBrush("TextMutedBrush", theme.TextMuted);
        SetBrush("AccentBrush", theme.Accent);
        SetBrush("AccentHoverBrush", theme.AccentHover);
        SetBrush("InfoBrush", theme.Info);
        SetBrush("OverlayBrush", theme.Overlay);
    }

    private static void SetBrush(string key, string color) =>
        Application.Current!.Resources[key] = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse(color));


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

            if (Brands.Count > 0) SelectedBrand = Brands[0];

            StatusMessage = "Odaberi brand i model automobila.";
        }
        catch (Exception ex)
        {
            HasError      = true;
            StatusMessage = $"Greška pri učitavanju opcija: {ex.Message}";
        }
    }

    private async Task LoadModelsForBrandAsync(string brand)
    {
        IsLoadingModels = true;
        StatusMessage   = $"Učitavam modele za {brand}…";

        try
        {
            var models = await _api.GetModelsForBrandAsync(brand);
            CarModels.Clear();
            foreach (var m in models) CarModels.Add(m);

            SelectedCarModel = CarModels.Count > 0 ? CarModels[0] : string.Empty;
            StatusMessage    = CarModels.Count > 0
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

    private async Task LoadStatsAsync()
    {
        try
        {
            var stats = await _api.GetStatsAsync();
            if (stats is null) return;

            StatsCards.Clear();
            StatsCards.Add(new StatCard("MAPE",       FormatPercent(stats.Mape),               ToneFromMape(stats.Mape)));
            StatsCards.Add(new StatCard("R²",         FormatNumber(stats.R2, 4),               "#38BDF8"));
            StatsCards.Add(new StatCard("MAE",        FormatCurrency(stats.Mae),               "#F97316"));
            StatsCards.Add(new StatCard("RMSE",       FormatCurrency(stats.Rmse),              "#F59E0B"));
            StatsCards.Add(new StatCard("Train MAPE", FormatPercent(stats.TrainMape),          "#22C55E"));
            StatsCards.Add(new StatCard("Uzorci",     FormatCount(stats.TrainingSamples),      "#A7F3D0"));

            StatsModelType = string.IsNullOrWhiteSpace(stats.BestModel)
                ? stats.ModelType
                : $"{stats.BestModel} (best)";
        }
        catch
        {
            // Statistika je opcionalna – tiho ignoriramo
        }
    }

    private static void Populate(ObservableCollection<string> target, IEnumerable<string> source)
    {
        target.Clear();
        foreach (var item in source) target.Add(item);
    }

    private static string FormatPercent(double? v)  => v is null ? "—" : $"{v:0.00}%";
    private static string FormatNumber(double? v, int d) => v is null ? "—" : v.Value.ToString($"0.{new string('0', d)}");
    private static string FormatCurrency(double? v) => v is null ? "—" : $"${v:0,0}";
    private static string FormatCount(int? v)        => v is null ? "—" : v.Value.ToString("0,0");

    private static string ToneFromMape(double? mape)
    {
        if (mape is null) return "#94A3B8";
        if (mape <= 10)   return "#22C55E";
        if (mape <= 15)   return "#F59E0B";
        return "#EF4444";
    }
}

public sealed record StatCard(string Label, string Value, string Accent);

public sealed record ThemeOption(
    string Name,
    ThemeVariant Variant,
    string Background,
    string Card,
    string CardInner,
    string Border,
    string Text,
    string TextSoft,
    string TextMuted,
    string Accent,
    string AccentHover,
    string Info,
    string Overlay)
{
    public override string ToString() => Name;
}
