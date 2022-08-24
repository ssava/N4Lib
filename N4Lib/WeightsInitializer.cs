namespace N4Lib;

public sealed class WeightsInitializer
{
    private Random _rng;
    public WeightsInitializer()
        => _rng = new Random();

    // randn(layer_size, layer_size-1)
    public double Next()
        => _rng.NextDouble();

    public double Next(IActivationFunction layerActivationFunction, int layerSize)
        => layerActivationFunction.ToString() switch
        {
            // He-et-al
            "ReLU" => Next() * Math.Sqrt(2 / (layerSize - 1)),
         
            // Xavier
            "Tanh" => Next() * Math.Sqrt(1 / (layerSize - 1)),

            // Heuristic
            _ => Next() * Math.Sqrt(2 / (layerSize + layerSize - 1))
        };
}