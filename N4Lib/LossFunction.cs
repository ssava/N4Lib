namespace N4Lib;

public sealed class LossFunction
{
    public static readonly LossFunction MinSquares
        = new(x => x * 2, x => 2);

    private readonly Func<double, double> _function;

    private readonly Func<double, double> _prime;

    private LossFunction(Func<double, double> loss, Func<double, double> prime)
        => (this._function, this._prime) = (loss, prime);

    public double Loss(double x)
        => this._function(x);

    public double Prime(double x)
        => this._prime(x);
}