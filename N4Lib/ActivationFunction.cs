namespace N4Lib;

public sealed record ActivationFunction : IActivationFunction
{
	private sealed class WeightOptimizers
	{
		public static readonly Func<double, double> HeuristingOptimization
			= layerSize => Math.Sqrt(2 / (layerSize + layerSize - 1));

		public static readonly Func<double, double> Xavier
			= layerSize => Math.Sqrt(1 / (layerSize - 1));

		public static readonly Func<double, double> HeEtAl
			= layerSize => Math.Sqrt(2 / (layerSize - 1));
	}

    public static readonly ActivationFunction Identity
		= new(x => x, x => 1);

    public static readonly ActivationFunction LeakyReLU
		= new(x => x * (x < 0 ? 0.01 : 1), x => x < 0 ? 0.01 : 1);

    public static readonly ActivationFunction None
              = new(x => x, x => x);

    public static readonly ActivationFunction ReLU
              = new(x => x < 0 ? 0 : x, x => x < 0 ? 0 : 1, WeightOptimizers.HeEtAl);

    public static readonly ActivationFunction Sigmoid
		= new(
				func: x => 1 / (1 + Math.Exp(-x)),
				prime: x => (1 / (1 + Math.Exp(-x)) * (1 - (1 / (1 + Math.Exp(-x)))))
		);

    public static readonly ActivationFunction SoftPlus
		= new(x => Math.Log(1 + Math.Exp(x)), x => 1 / (1 + Math.Exp(-x)));

    public static readonly ActivationFunction SoftSign
		= new(
			func: x => x / (1 + Math.Abs(x)),
			prime: x => 1 / Math.Pow(1 + Math.Abs(x), 2)
		);

    public static readonly ActivationFunction Tanh
		= new(x => Math.Tanh(x), x => Math.Pow(Math.Tanh(x), 2), WeightOptimizers.Xavier);

    private Func<double, double> _activate { get; }

    private Func<double, double> _prime { get; }

	private Func<double, double> _weightOptimization { get; }

    private ActivationFunction
        (Func<double, double> func, Func<double, double> prime)
		: this(func, prime, WeightOptimizers.HeuristingOptimization)
	{
	}

	private ActivationFunction
        (Func<double, double> func,
		 Func<double, double> prime,
		 Func<double, double> weightOptimizer)
	{
		_activate = func;
		_prime = prime;
		_weightOptimization = weightOptimizer;
	}

	public double Activate(double x)
		=> this._activate(x);

	public double Prime(double x)
		=> this._prime(x);

	public double OptimizationFactor(double x)
		=> this._weightOptimization(x);
}
