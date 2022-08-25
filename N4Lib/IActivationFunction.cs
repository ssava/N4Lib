namespace N4Lib;

public interface IActivationFunction
{
    double Activate(double x);
    double Prime(double x);
    double OptimizationFactor(double x);
}