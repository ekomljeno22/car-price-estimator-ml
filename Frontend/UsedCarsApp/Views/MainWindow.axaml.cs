using Avalonia.Controls;
using UsedCarsApp.ViewModels;
using UsedCarsApp.Services;

namespace UsedCarsApp.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

        DataContext = new MainViewModel(new CarApiService("http://localhost:5000"));

        Opened += (_, _) => Opacity = 1;
    }
}