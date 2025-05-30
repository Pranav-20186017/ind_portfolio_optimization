import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

// Re‑usable equation wrapper for consistent styling
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const V2RatioPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>V2 Ratio for Indian Market Strategies | QuantPort India Docs</title>
        <meta
          name="description"
          content="Evaluate Indian investment strategies with the V2 Ratio. Compare growth rates to drawdown volatility for NSE/BSE portfolios relative to Indian market benchmarks."
        />
        <meta property="og:title" content="V2 Ratio for Indian Market Strategies | QuantPort India Docs" />
        <meta property="og:description" content="Evaluate Indian investment strategies with the V2 Ratio. Compare growth rates to drawdown volatility for NSE/BSE portfolios relative to Indian market benchmarks." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/v2-ratio" />
      </Head>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Docs</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* Title */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            V2 Ratio
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A performance measure that evaluates relative growth rate compared to benchmark divided by relative drawdown volatility
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            <strong>V2 Ratio</strong> is an advanced performance measure developed to provide a more comprehensive assessment of investment strategy performance relative to a benchmark. Unlike traditional metrics like the Sharpe Ratio that focus solely on return-to-volatility relationships, the V2 Ratio examines both growth characteristics and drawdown behavior.
          </Typography>
          <Typography paragraph>
            Introduced by researchers seeking more robust evaluation methods, the V2 Ratio compares two critical dimensions of performance: the relative growth rate of an investment strategy compared to its benchmark, and the relative drawdown volatility between the strategy and the benchmark. This dual focus allows for a more nuanced evaluation of risk-adjusted performance that better accounts for investor preferences and real-world portfolio behavior.
          </Typography>
          <Typography paragraph>
            By incorporating drawdown volatility rather than just standard deviation, the V2 Ratio acknowledges that investors are typically more concerned with downside risk and capital preservation than with symmetric risk measures. This makes it particularly valuable for evaluating strategies designed for long-term wealth accumulation where avoiding significant drawdowns is crucial.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Imagine you're comparing two different routes to the same destination. The first consideration is how much faster each route is compared to the standard highway (relative speed). The second consideration is how consistent the traffic flow is compared to the highway's traffic patterns (relative smoothness).
          </Typography>
          <Typography paragraph>
            The V2 Ratio is like dividing your relative speed advantage by your relative traffic variability. A high V2 Ratio means you're getting substantially better progress with proportionally less stop-and-start traffic compared to the standard route.
          </Typography>
          <Typography paragraph>
            For investments, instead of measuring just how much better your returns are than the benchmark, the V2 Ratio asks: "Are the superior returns worth the potentially different pattern of ups and downs?" It's not just about beating the benchmark; it's about whether you're beating it in a way that delivers a smoother journey for investors.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Financial analogy:</strong> If the Sharpe Ratio compares your investment's "miles per gallon" to the risk-free rate, and the Information Ratio compares your "miles per gallon" to a benchmark index, the V2 Ratio is like comparing your "average speed relative to a benchmark" divided by your "relative bumpiness of the ride." This helps determine if your investment strategy is providing meaningfully better progress toward the destination with proportionally less discomfort along the way.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            The V2 Ratio is formally defined as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>V2 Ratio Formula</strong></Typography>
            <Equation math="\text{V2 Ratio} = \frac{\text{Relative Growth Rate}}{\text{Relative Drawdown Volatility}}" />
            <Typography variant="body2">
              where the numerator measures outperformance in growth rate and the denominator measures the relative volatility of drawdowns.
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Relative Growth Rate</Typography>
          <Typography paragraph>
            The relative growth rate compares the compound annual growth rate (CAGR) of the strategy to that of the benchmark:
          </Typography>

          <Equation math="\text{Relative Growth Rate} = \frac{\text{CAGR}_{\text{strategy}}}{\text{CAGR}_{\text{benchmark}}}" />

          <Typography paragraph>
            The CAGR for each is calculated as:
          </Typography>

          <Equation math="\text{CAGR} = \left(\frac{P_{\text{final}}}{P_{\text{initial}}}\right)^{\frac{1}{T}} - 1" />

          <Typography paragraph>
            where <InlineMath math="P_{\text{final}}" /> is the final portfolio value, <InlineMath math="P_{\text{initial}}" /> is the initial portfolio value, and <InlineMath math="T" /> is the time period in years.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Relative Drawdown Volatility</Typography>
          <Typography paragraph>
            The relative drawdown volatility compares the severity and frequency of drawdowns between the strategy and the benchmark:
          </Typography>

          <Equation math="\text{Relative Drawdown Volatility} = \frac{\sigma_{\text{DD,strategy}}}{\sigma_{\text{DD,benchmark}}}" />

          <Typography paragraph>
            Where <InlineMath math="\sigma_{\text{DD}}" /> is the standard deviation of drawdown magnitudes over the evaluation period for each series. The drawdown at time t is defined as:
          </Typography>

          <Equation math="\text{DD}_t = \frac{P_t - P_{\text{max}}}{P_{\text{max}}}" />

          <Typography paragraph>
            with <InlineMath math="P_{\text{max}}" /> representing the maximum portfolio value achieved up to time t.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Interpretation of V2 Ratio Values</Typography>
          <Typography paragraph>
            The interpretation of the V2 Ratio follows these guidelines:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                                <strong>V2 Ratio {'>'}  1:</strong> The strategy delivers superior growth relative to drawdown risk compared to the benchmark.              </Typography>            </li>            <li>              <Typography paragraph>                <strong>V2 Ratio = 1:</strong> The strategy's risk-adjusted performance is equivalent to the benchmark.              </Typography>            </li>            <li>              <Typography paragraph>                <strong>V2 Ratio {'<'} 1:</strong> The strategy underperforms the benchmark when accounting for drawdown volatility.
              </Typography>
            </li>
          </ul>

          <Typography paragraph>
            A higher V2 Ratio indicates that the strategy delivers more efficient growth per unit of drawdown volatility relative to the benchmark, which is particularly meaningful for investors with longer time horizons and sensitivity to significant market declines.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Alternative Formulation</Typography>
          <Typography paragraph>
            In some implementations, the V2 Ratio may be calculated using a modified approach:
          </Typography>

          <Equation math="\text{V2 Ratio}_{\text{alt}} = \frac{\text{CAGR}_{\text{strategy}} - \text{CAGR}_{\text{benchmark}}}{\sigma_{\text{DD,strategy}} - \sigma_{\text{DD,benchmark}}}" />

          <Typography paragraph>
            This formulation focuses on the absolute differences rather than the ratios, which can be more appropriate when dealing with negative growth rates or when the benchmark has very low drawdown volatility.
          </Typography>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Portfolio Analysis</Typography>
          <Typography paragraph>
            In our portfolio optimization service, we calculate the V2 Ratio through the following steps:
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                <strong>Data Collection:</strong> Time series of both strategy and benchmark returns are collected over the same time period, with sufficient history to capture various market conditions.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Growth Rate Calculation:</strong> The CAGR is computed for both the strategy and the benchmark to determine their respective annualized growth rates.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Drawdown Analysis:</strong> The drawdown sequences for both the strategy and benchmark are calculated at regular intervals (typically daily or monthly).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Volatility Measurement:</strong> The standard deviation of drawdown magnitudes is calculated for both series to quantify drawdown volatility.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Ratio Computation:</strong> The relative growth rate is divided by the relative drawdown volatility to produce the V2 Ratio.
              </Typography>
            </li>
          </ol>
          <Typography paragraph>
            This implementation allows for the comparison of strategies across different asset classes, time periods, and market regimes. The V2 Ratio can be particularly useful for:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Strategy Selection:</strong> Identifying strategies that provide the most efficient growth per unit of drawdown risk.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Manager Evaluation:</strong> Assessing investment managers based on their ability to generate superior growth rates while controlling drawdown volatility.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Allocation:</strong> Determining optimal capital allocation between different strategies based on their V2 Ratios.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Retirement Planning:</strong> Evaluating investment approaches for long-term wealth accumulation where drawdown management is crucial.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's calculate the V2 Ratio for a hypothetical investment strategy compared to a benchmark index over a 5-year period:
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: Growth Rate Calculation</Typography>
          <Typography paragraph>
            Assume the following end-of-year values for a $100,000 initial investment:
          </Typography>
          <Grid container spacing={3} sx={{ mb: 2 }}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom><strong>Strategy Performance</strong></Typography>
                <ul>
                  <li>Initial value: $100,000</li>
                  <li>Year 1: $112,000</li>
                  <li>Year 2: $126,560</li>
                  <li>Year 3: $120,232</li>
                  <li>Year 4: $135,862</li>
                  <li>Year 5: $156,241</li>
                </ul>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom><strong>Benchmark Performance</strong></Typography>
                <ul>
                  <li>Initial value: $100,000</li>
                  <li>Year 1: $108,000</li>
                  <li>Year 2: $114,480</li>
                  <li>Year 3: $109,900</li>
                  <li>Year 4: $118,692</li>
                  <li>Year 5: $130,561</li>
                </ul>
              </Box>
            </Grid>
          </Grid>

          <Typography paragraph>
            Calculate the CAGR for both series:
          </Typography>

          <Typography paragraph>
            <strong>Strategy CAGR:</strong> <InlineMath math="(156,241 / 100,000)^{1/5} - 1 = 0.0933" /> or 9.33% per year
          </Typography>

          <Typography paragraph>
            <strong>Benchmark CAGR:</strong> <InlineMath math="(130,561 / 100,000)^{1/5} - 1 = 0.0548" /> or 5.48% per year
          </Typography>

          <Typography paragraph>
            <strong>Relative Growth Rate:</strong> <InlineMath math="0.0933 / 0.0548 = 1.70" />
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Drawdown Analysis</Typography>
          <Typography paragraph>
            From the annual data provided, we can identify the maximum drawdowns in each series:
          </Typography>

          <Typography paragraph>
            <strong>Strategy Drawdowns:</strong> Year 2-3: (120,232 - 126,560) / 126,560 = -5.00%
          </Typography>

          <Typography paragraph>
            <strong>Benchmark Drawdowns:</strong> Year 2-3: (109,900 - 114,480) / 114,480 = -4.00%
          </Typography>

          <Typography paragraph>
            For a more comprehensive analysis, we would typically use more frequent data points (e.g., monthly or daily). Let's assume we've analyzed the intra-year movements and determined the following drawdown statistics:
          </Typography>

          <Grid container spacing={3} sx={{ mb: 2 }}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom><strong>Strategy Drawdowns</strong></Typography>
                <ul>
                  <li>Number of drawdowns: 8</li>
                  <li>Average drawdown: -3.20%</li>
                  <li>Standard deviation of drawdowns: 2.10%</li>
                </ul>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom><strong>Benchmark Drawdowns</strong></Typography>
                <ul>
                  <li>Number of drawdowns: 6</li>
                  <li>Average drawdown: -2.80%</li>
                  <li>Standard deviation of drawdowns: 1.50%</li>
                </ul>
              </Box>
            </Grid>
          </Grid>

          <Typography paragraph>
            <strong>Relative Drawdown Volatility:</strong> <InlineMath math="2.10\% / 1.50\% = 1.40" />
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Calculate the V2 Ratio</Typography>
          <Typography paragraph>
            <strong>V2 Ratio:</strong> <InlineMath math="\text{Relative Growth Rate} / \text{Relative Drawdown Volatility} = 1.70 / 1.40 = 1.21" />
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 4: Interpretation</Typography>
          <Typography paragraph>
            A V2 Ratio of 1.21 indicates that the strategy delivers 21% more growth per unit of drawdown volatility compared to the benchmark. This suggests that the strategy's superior returns (9.33% vs 5.48%) more than compensate for its higher drawdown volatility (2.10% vs 1.50%).
          </Typography>

          <Typography paragraph>
            If the ratio had been below 1.0, it would suggest that the additional growth did not adequately compensate for the increased drawdown risk compared to the benchmark.
          </Typography>
        </Paper>

        {/* Practical Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          <Typography paragraph>
            The V2 Ratio serves several important practical purposes in investment analysis and decision-making:
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Strategy Evaluation and Selection</Typography>
          <Typography paragraph>
            The V2 Ratio provides a framework for comparing different investment strategies by explicitly considering both their growth potential and drawdown characteristics. This is particularly valuable when evaluating strategies for long-term wealth accumulation, where both the rate of compounding and the severity of setbacks significantly impact final outcomes.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Retirement Planning</Typography>
          <Typography paragraph>
            For retirement planning, where sequence of returns risk becomes critical, the V2 Ratio helps identify strategies that achieve growth targets while minimizing the potential for severe drawdowns that could compromise financial security. This is especially important for investors approaching or in retirement, who have limited ability to recover from significant market declines.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Risk Management Framework</Typography>
          <Typography paragraph>
            The V2 Ratio can serve as part of a comprehensive risk management framework by encouraging portfolio managers to focus not just on return generation but also on controlling drawdown patterns. By highlighting the relationship between growth and drawdown volatility, it promotes a more balanced approach to risk-taking.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Performance Attribution</Typography>
          <Typography paragraph>
            Using the V2 Ratio in performance attribution helps identify whether outperformance comes from genuinely superior risk management or simply from taking more risk during favorable market conditions. This distinction is crucial for determining whether manager skill or market exposure is driving results.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Client Communication</Typography>
          <Typography paragraph>
            The V2 Ratio provides a useful tool for explaining investment strategy performance to clients in terms they can understand. Rather than focusing solely on abstract metrics like standard deviation, it frames risk in terms of drawdowns, which directly connect to clients' experience of portfolio volatility and their emotional reactions to market declines.
          </Typography>
        </Paper>

        {/* Advantages and Limitations */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Advantages and Limitations
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="primary">
                  Advantages
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Investor-centric perspective:</strong> Focuses on drawdowns, which align closely with how investors actually experience and react to risk in their portfolios.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Growth orientation:</strong> Emphasizes compound growth rates rather than arithmetic returns, better reflecting the actual wealth creation process over time.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Benchmark relevance:</strong> Directly compares strategy performance to a specific benchmark rather than to an abstract concept like the risk-free rate.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Time horizon flexibility:</strong> Can be applied across various time horizons, making it suitable for both tactical and strategic performance evaluation.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Intuitive interpretation:</strong> Provides a straightforward measure of whether a strategy delivers enough additional growth to compensate for any additional drawdown volatility.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Limitations
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Data requirements:</strong> Requires sufficient historical data to capture meaningful drawdown patterns, which may not be available for newer strategies or funds.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Benchmark sensitivity:</strong> Results are highly dependent on the choice of benchmark, potentially leading to different conclusions when different reference indices are used.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Potential distortions:</strong> In cases where the benchmark has very low drawdown volatility, the relative measure may become disproportionately large.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Limited standardization:</strong> As a relatively newer metric, the V2 Ratio lacks the widespread adoption and standardization of more established measures like the Sharpe or Information Ratios.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Backward-looking nature:</strong> Like most performance metrics, the V2 Ratio is based on historical data and may not reliably predict future performance patterns.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>References</Typography>
          <ul>
            <li>
              <Typography paragraph>
                Caporin, M., & Lisi, F. (2011). "Comparing and selecting performance measures using rank correlations." Economics: The Open-Access, Open-Assessment E-Journal, 5.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Eling, M., & Schuhmacher, F. (2007). "Does the choice of performance measure influence the evaluation of hedge funds?" Journal of Banking & Finance, 31(9), 2632-2647.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Bacon, C. R. (2013). "Practical Risk-Adjusted Performance Measurement." Wiley Finance.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Cogneau, P., & Hübner, G. (2009). "The (more than) 100 ways to measure portfolio performance." Journal of Performance Measurement, 13(4), 56-71.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Schuhmacher, F., & Eling, M. (2012). "A decision-theoretic foundation for reward-to-risk performance measures." Journal of Banking & Finance, 36(7), 2077-2082.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Related Topics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Information Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A performance metric that evaluates active return per unit of risk relative to a benchmark index.
                </Typography>
                <Link href="/docs/information-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Maximum Drawdown
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure of the largest peak-to-trough decline in a portfolio's value, representing the worst-case scenario for an investment.
                </Typography>
                <Link href="/docs/maximum-drawdown" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Conditional Drawdown at Risk (CDaR)
                </Typography>
                <Typography variant="body2" paragraph>
                  The expected value of drawdown when exceeding the Drawdown at Risk threshold, measuring tail drawdown risk.
                </Typography>
                <Link href="/docs/cdar" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </>
  );
};

// Add getStaticProps to generate this page at build time
export const getStaticProps = async () => {
  return {
    props: {},
  };
};

export default V2RatioPage; 