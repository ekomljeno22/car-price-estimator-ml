using UsedCarsApi.Services;

var builder = WebApplication.CreateBuilder(args);

// ── Services ──────────────────────────────────────────────────────────────────
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// CORS – allow Avalonia desktop app (localhost) and any dev origin
builder.Services.AddCors(o => o.AddDefaultPolicy(p =>
    p.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod()));

// Typed HttpClient pointing at the Python ML service
builder.Services.AddHttpClient<ICarPredictionService, CarPredictionService>(client =>
{
    var mlUrl = builder.Configuration["MlService:BaseUrl"] ?? "http://localhost:8000";
    client.BaseAddress = new Uri(mlUrl);
    client.Timeout     = TimeSpan.FromSeconds(30);
});

var app = builder.Build();

// ── Middleware ────────────────────────────────────────────────────────────────
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseCors();
app.MapControllers();

app.Run();