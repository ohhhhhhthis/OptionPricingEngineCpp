// #pragma once
#ifndef Pricing_Engine_hpp
#define Pricing_Engine_hpp

#include <ostream>
#include <cmath>
#include <optional>
#include <random>
#include <chrono>
#include <Eigen/Dense>

struct MonteCarloResult
{
    int sample_size = 0;
    double price = 0.0;
    double standard_error = 0.0;
    double lower_bound = 0.0;
    double upper_bound = 0.0;
    double elapsed_time = 0.0;

    MonteCarloResult(const Eigen::ArrayXd &payoff, const std::chrono::duration<double> &duration, const double discount = 1.0)
    {
        sample_size = payoff.size();
        double raw_mean = payoff.mean();
        double raw_var = (payoff - raw_mean).square().sum() / (sample_size - 1);
        price = discount * raw_mean;
        standard_error = discount * sqrt(raw_var / sample_size);
        lower_bound = price - 1.96 * standard_error;
        upper_bound = price + 1.96 * standard_error;
        elapsed_time = duration.count();
    }

    friend std::ostream &operator<<(std::ostream &os, const MonteCarloResult &res)
    {
        os << "Sample size        = " << res.sample_size
           << "\nOption price     = " << res.price
           << "\nStandard error   = " << res.standard_error
           << "\n95% CI           = [" << res.lower_bound << ", " << res.upper_bound << "]\n"
           << "Simulation time    = " << res.elapsed_time << "s\n";
        return os;
    }
};

enum OptionType : int
{
    Call = 1,
    Put = -1
};

class OptionPricer
{
public:
    OptionPricer(double S0, double K, double T, double r, double q, double sigma, OptionType flag, int seed = 42) : S0(S0), K(K), T(T), r(r), q(q), sigma(sigma), flag(flag), seed(seed), gen(seed) {};

    double phi() { return static_cast<double>(flag); }
    void resetSeed() { gen.seed(seed); };
    void resetSeed(int new_seed) { seed = new_seed, gen.seed(new_seed); };
    void weighted_Laguerre(const Eigen::Ref<const Eigen::ArrayXd> &s, Eigen::Ref<Eigen::MatrixXd> X, int l);

    double BSM_European(double spot = -1.0, double maturity = -1.0);
    double BSM_Continuous_Geometric_Asian();
    double BSM_Discrete_Geometric_Asian(int m);
    double BSM_Moment_Matching_Arithmatic_Asian(int m);
    double CRR_European(int m);
    double CRR_American(int m, bool bs);
    double BBSR_American(int m);

    Eigen::ArrayXXd Brownian_Generator(int n, int m, bool MomentMatch = false);
    std::pair<Eigen::ArrayXXd, std::optional<Eigen::ArrayXXd>>
    Path_Simulation(int n, int m, bool Antithetic = false, bool EMS = false, bool MomentMatch = false);
    MonteCarloResult Monte_Carlo_European(int n, bool Antithetic = false, bool EMS = false, bool MomentMatch = false);
    MonteCarloResult Monte_Carlo_Asian(int n, int m, bool Antithetic = false, bool EMS = false, bool MomentMatch = false, bool CtrlVar = false);
    MonteCarloResult Least_Squares_Monte_Carlo_American(int n, int m, int l);

private:
    double S0;
    double K;
    double T;
    double r;
    double q;
    double sigma;
    OptionType flag;
    int seed;
    std::mt19937_64 gen;
};

#endif