using UsedCarsApp.Services;
using UsedCarsApp.ViewModels;

namespace UsedCarsApp.ViewModels;

public sealed class ViewModelLocator
{
    private static readonly ICarApiService _apiService =
        new CarApiService("http://localhost:5000");

    public MainViewModel Main { get; } = new(_apiService);
}