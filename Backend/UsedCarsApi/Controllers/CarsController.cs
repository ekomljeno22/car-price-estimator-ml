using Microsoft.AspNetCore.Mvc;
using UsedCarsApi.Models;
using UsedCarsApi.Services;

namespace UsedCarsApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class CarsController(ICarPredictionService svc) : ControllerBase
{
    /// <summary>Predict the price of a used car.</summary>
    [HttpPost("predict")]
    [ProducesResponseType<PredictResponse>(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status502BadGateway)]
    public async Task<IActionResult> Predict([FromBody] PredictRequest request, CancellationToken ct)
    {
        var result = await svc.PredictAsync(request, ct);
        if (result is null)
            return StatusCode(502, "ML service returned an empty response.");

        return Ok(result);
    }

    /// <summary>Get valid dropdown option values for the UI.</summary>
    [HttpGet("options")]
    [ProducesResponseType<CarOptions>(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status502BadGateway)]
    public async Task<IActionResult> Options(CancellationToken ct)
    {
        var options = await svc.GetOptionsAsync(ct);
        if (options is null)
            return StatusCode(502, "ML service returned an empty response.");

        return Ok(options);
    }

    /// <summary>Health check — also pings the ML service.</summary>
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