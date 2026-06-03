using UsedCarsApp.Services;
using UsedCarsApp.ViewModels;

namespace UsedCarsApp.ViewModels;

/// <summary>
/// Simple service-locator / composition root used by App.axaml DataContext.
/// For larger apps swap this out for Microsoft.Extensions.DependencyInjection.
/// </summary>
public sealed class ViewModelLocator
{
    private static readonly ICarApiService _apiService =
        new CarApiService("http://localhost:5000");

    public MainViewModel Main { get; } = new(_apiService);
}