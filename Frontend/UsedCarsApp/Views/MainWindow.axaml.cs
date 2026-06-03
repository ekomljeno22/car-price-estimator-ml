using Avalonia.Controls;
using UsedCarsApp.ViewModels;
using UsedCarsApp.Services;

namespace UsedCarsApp.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

        // Wire the DataContext to the real ViewModel with the API service
        DataContext = new MainViewModel(new CarApiService("http://localhost:5000"));
    }
}