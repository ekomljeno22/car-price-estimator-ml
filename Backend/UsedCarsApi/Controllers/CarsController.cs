using Microsoft.AspNetCore.Mvc;
using UsedCarsApi.Models;
using UsedCarsApi.Services;

namespace UsedCarsApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class CarsController(ICarPredictionService svc) : ControllerBase
{
    [HttpPost("predict")]
    [ProducesResponseType<PredictResponse>(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status502BadGateway)]
    public async Task<IActionResult> Predict([FromBody] PredictRequest request, CancellationToken ct)
    {
        var result = await svc.PredictAsync(request, ct);
        if (result is null) return StatusCode(502, "ML service returned an empty response.");
        return Ok(result);
    }

    [HttpGet("options")]
    [ProducesResponseType<CarOptions>(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status502BadGateway)]
    public async Task<IActionResult> Options(CancellationToken ct)
    {
        var options = await svc.GetOptionsAsync(ct);
        if (options is null) return StatusCode(502, "ML service returned an empty response.");
        return Ok(options);
    }

    [HttpGet("models-for-brand/{brand}")]
    [ProducesResponseType<BrandModelsResponse>(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status502BadGateway)]
    public async Task<IActionResult> ModelsForBrand(string brand, CancellationToken ct)
    {
        try
        {
            var result = await svc.GetModelsForBrandAsync(brand, ct);
            if (result is null) return StatusCode(502, "ML service returned an empty response.");
            return Ok(result);
        }
        catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.NotFound)
        {
            return NotFound($"Brand '{brand}' not found.");
        }
    }

    [HttpGet("stats")]
    [ProducesResponseType<ModelStats>(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status502BadGateway)]
    public async Task<IActionResult> Stats(CancellationToken ct)
    {
        var stats = await svc.GetStatsAsync(ct);
        if (stats is null) return StatusCode(502, "ML service returned an empty response.");
        return Ok(stats);
    }


    [HttpGet("charts")]
    [ProducesResponseType<ChartsResponse>(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status502BadGateway)]
    public async Task<IActionResult> Charts(CancellationToken ct)
    {
        var result = await svc.GetChartsAsync(ct);
        if (result is null) return StatusCode(502, "ML service returned an empty response.");

        var rewrittenCharts = result.Charts
            .Select(c => c with { Url = $"/api/cars/charts/{Uri.EscapeDataString(c.Filename)}" })
            .ToList();

        return Ok(new ChartsResponse(rewrittenCharts));
    }

    [HttpGet("charts/{filename}")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status502BadGateway)]
    public async Task<IActionResult> ChartImage(string filename, CancellationToken ct)
    {
        if (filename.Contains('/') || filename.Contains('\\') || filename.Contains(".."))
            return BadRequest("Neispravan naziv datoteke.");

        try
        {
            var bytes = await svc.GetChartImageAsync(filename, ct);
            if (bytes is null or { Length: 0 })
                return StatusCode(502, "ML service returned empty image.");

            return File(bytes, "image/png", filename);
        }
        catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.NotFound)
        {
            return NotFound($"Graf '{filename}' nije pronađen.");
        }
    }

    [HttpGet("health")]
    public async Task<IActionResult> Health(CancellationToken ct)
    {
        try
        {
            var options = await svc.GetOptionsAsync(ct);
            return Ok(new { status = "ok", ml_reachable = options is not null });
        }
        catch
        {
            return Ok(new { status = "ok", ml_reachable = false });
        }
    }
}