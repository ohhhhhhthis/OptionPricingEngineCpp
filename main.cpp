#include "PricingEngine.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
// #include <fstream>
using namespace std;

void printSeparator(int width = 90)
{
    cout << string(width, '-') << "\n";
}

int main()
{
    // std::ofstream outFile("output.txt");
    // std::streambuf* coutBuf = std::cout.rdbuf();
    // std::cout.rdbuf(outFile.rdbuf());

    cout << fixed << setprecision(3);

    cout << "\n=========================================================================\n";
    cout << "  Part 1: European Vanilla Put Option (Monte Carlo, Antithetic Approach)\n";
    cout << "=========================================================================\n";
    {
        double S0 = 80.0, K = 100.0, T = 0.5, r = 0.04, q = 0.02, sigma = 0.2;
        OptionType flag = OptionType::Put;
        OptionPricer pricer(S0, K, T, r, q, sigma, flag);

        double exact_bsm = pricer.BSM_European();
        cout << "Exact BSM Price: " << exact_bsm << "\n\n";

        vector<int> sample_sizes = {1000, 5000, 10000, 50000, 100000, 500000, 1000000};

        cout << setw(10) << "N" << setw(13) << "Price (Std)" << setw(12) << "StdErr" << setw(13) << "Abs Error" << setw(24) << "95% CI" << setw(12) << "Time (s)" << "\n";
        printSeparator(84);

        for (int N : sample_sizes)
        {
            pricer.resetSeed();
            auto res = pricer.Monte_Carlo_European(N, false, false, false);
            double abs_err = std::abs(res.price - exact_bsm);

            string ci = "[" + to_string(res.lower_bound).substr(0, 5) + ", " + to_string(res.upper_bound).substr(0, 5) + "]";
            cout << setw(10) << N << setw(13) << res.price << setw(12) << res.standard_error << setw(13) << abs_err << setw(24) << ci << setw(12) << res.elapsed_time << "\n";
        }

        cout << "\n---  With Antithetic Variates ---\n";
        cout << setw(10) << "N" << setw(13) << "Price (Anti)" << setw(12) << "StdErr" << setw(13) << "Abs Error" << setw(16) << "Var Reduction" << setw(12) << "Time (s)" << "\n";
        printSeparator(76);

        for (int N : sample_sizes)
        {
            pricer.resetSeed();
            auto res_std = pricer.Monte_Carlo_European(N, false, false, false);
            pricer.resetSeed();
            auto res_anti = pricer.Monte_Carlo_European(N, true, false, false);

            double abs_err = abs(res_anti.price - exact_bsm);
            double var_reduction = 1.0 - (res_anti.standard_error * res_anti.standard_error) / (res_std.standard_error * res_std.standard_error);

            cout << setw(10) << N << setw(13) << res_anti.price << setw(12) << res_anti.standard_error << setw(13) << abs_err << setw(15) << (var_reduction * 100.0) << "%" << setw(12) << res_anti.elapsed_time << "\n";
        }
    }

    cout << "\n=========================================================================\n";
    cout << " Part 2: Asian Call Option (Monte Carlo, Control Variate, Moment Match)\n";
    cout << "=========================================================================\n";
    {
        double S0 = 120.0, K = 100.0, T = 1.0, r = 0.10, q = 0.0, sigma = 0.20;
        int m = 50;
        OptionType flag = OptionType::Call;
        OptionPricer pricer(S0, K, T, r, q, sigma, flag);

        double mm_price = pricer.BSM_Moment_Matching_Arithmatic_Asian(m);
        double geo_exact = pricer.BSM_Discrete_Geometric_Asian(m);
        cout << "Analytic Discrete Geometric Asian Price   : " << geo_exact << "\n";
        cout << "Log-normal Moment Matching Price          : " << mm_price << "\n\n";

        vector<int> sample_sizes = {1000, 5000, 10000, 50000, 100000};

        cout << setw(8) << "N" << setw(13) << "MC Standard" << setw(10) << "StdErr" << setw(13) << "MC CtrlVar" << setw(10) << "StdErr" << setw(16) << "Var Reduction" << setw(10) << "Time(CV)" << "\n";
        printSeparator(80);

        for (int N : sample_sizes)
        {
            pricer.resetSeed();
            auto res_std = pricer.Monte_Carlo_Asian(N, m, false, false, false, false);
            pricer.resetSeed();
            auto res_cv = pricer.Monte_Carlo_Asian(N, m, false, false, false, true);

            double var_red = 1.0 - (res_cv.standard_error * res_cv.standard_error) /
                                       (res_std.standard_error * res_std.standard_error);

            cout << setw(8) << N << setw(13) << res_std.price << setw(10) << res_std.standard_error << setw(13) << res_cv.price << setw(10) << res_cv.standard_error << setw(15) << (var_red * 100.0) << "%" << setw(10) << res_cv.elapsed_time << "\n";
        }
    }

    cout << "\n=========================================================================\n";
    cout << "             Part 3: American Put Option (LSMC, BBSR)\n";
    cout << "=========================================================================\n";
    {
        double S0 = 100.0, K = 100.0, T = 1.0 / 12.0, r = 0.04, q = 0.02, sigma = 0.20;
        OptionType flag = OptionType::Put;
        OptionPricer pricer(S0, K, T, r, q, sigma, flag);

        double bbsr_benchmark = pricer.BBSR_American(1000);
        cout << "BBSR Price (m=1000): " << bbsr_benchmark << "\n\n";

        cout << "--- 1. Impact of Path Number (N) [M = 50, L = 3] ---\n";
        cout << setw(10) << "N" << setw(14) << "LSMC Price" << setw(14) << "Diff vs BBSR" << setw(12) << "Time (s)" << "\n";
        printSeparator(50);
        for (int N : {2000, 10000, 50000, 100000})
        {
            pricer.resetSeed();
            auto res = pricer.Least_Squares_Monte_Carlo_American(N, 50, 3);
            cout << setw(10) << N << setw(14) << res.price << setw(14) << (res.price - bbsr_benchmark) << setw(12) << res.elapsed_time << "\n";
        }

        cout << "\n--- 2. Impact of Time Steps (M) [N = 50000, L = 3] ---\n";
        cout << setw(10) << "M" << setw(14) << "LSMC Price" << setw(14) << "Diff vs BBSR" << setw(12) << "Time (s)" << "\n";
        printSeparator(50);
        for (int M : {10, 25, 50, 100, 200})
        {
            pricer.resetSeed();
            auto res = pricer.Least_Squares_Monte_Carlo_American(50000, M, 3);
            cout << setw(10) << M << setw(14) << res.price << setw(14) << (res.price - bbsr_benchmark) << setw(12) << res.elapsed_time << "\n";
        }

        cout << "\n--- 3. Impact of Regressors (L) [N = 50000, M = 50] ---\n";
        cout << setw(10) << "L" << setw(14) << "LSMC Price" << setw(14) << "Diff vs BBSR" << setw(12) << "Time (s)" << "\n";
        printSeparator(50);
        for (int L : {2, 3, 4, 5, 6})
        {
            pricer.resetSeed();
            auto res = pricer.Least_Squares_Monte_Carlo_American(50000, 50, L);
            cout << setw(10) << L << setw(14) << res.price << setw(14) << (res.price - bbsr_benchmark) << setw(12) << res.elapsed_time << "\n";
        }
    }

    return 0;
}