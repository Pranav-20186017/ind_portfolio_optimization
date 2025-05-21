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

const ModiglianiRiskAdjustedPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Modigliani Risk-Adjusted Performance (M²) | Portfolio Optimization</title>
        <meta
          name="description"
          content="Learn about Modigliani Risk-Adjusted Performance (M²), a measure that adjusts portfolio returns to match market volatility, allowing direct comparison with benchmark returns."
        />
      </Head>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/education" passHref>
            <Button variant="outlined" color="primary">← Back to Education</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* Title */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Modigliani Risk-Adjusted Performance (M²)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A measure that adjusts portfolio returns to match market volatility, allowing direct comparison with benchmark returns
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            <strong>Modigliani Risk-Adjusted Performance</strong> (also known as M², M-squared, or RAP) is an innovative risk-adjusted performance measure developed by Nobel Prize-winning economist Franco Modigliani and his granddaughter Leah Modigliani in 1997. Unlike abstract ratios like the Sharpe ratio, M² expresses risk-adjusted performance in percentage terms, making it exceptionally intuitive for investors to interpret and compare.
          </Typography>
          <Typography paragraph>
            The core concept behind M² is elegant: it adjusts a portfolio's returns to match the volatility level of some benchmark (typically the market), allowing for direct, apples-to-apples comparison between portfolio performance and benchmark performance. This is achieved by creating a theoretical leveraged or deleveraged version of the original portfolio that has the same risk level as the benchmark.
          </Typography>
          <Typography paragraph>
            M² solves a fundamental problem in performance measurement by translating the Sharpe ratio's dimensionless value into percentage returns that investors can immediately understand. For instance, an M² of 2% indicates that the portfolio, when adjusted to have the same risk as the benchmark, outperformed that benchmark by 2 percentage points—a clear and actionable insight.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Imagine you're comparing two different marathon runners with different strategies:
          </Typography>
          <Typography paragraph>
            <strong>Runner A</strong> sprints at full speed but takes frequent rest breaks when exhausted.
          </Typography>
          <Typography paragraph>
            <strong>Runner B</strong> maintains a consistent, moderate pace throughout the race.
          </Typography>
          <Typography paragraph>
            Both runners might complete the marathon in the same total time, but their approach to managing energy (risk) is completely different. To fairly compare their performance, you might ask: "If Runner A were forced to run with the same consistency (risk level) as Runner B, how would their finishing time compare?"
          </Typography>
          <Typography paragraph>
            That's essentially what M² does. It asks: "If we adjusted this portfolio to have exactly the same volatility as the market benchmark, what would its returns be?" This allows for direct comparison between different investment strategies, regardless of their original risk levels.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Financial analogy:</strong> M² is like normalizing cars' fuel efficiency at a standard speed of 55 mph. Even if one car was driven aggressively and another conservatively, adjusting both to the same "risk level" (speed) allows for fair comparison of their inherent efficiency. Similarly, M² adjusts portfolios to the same risk level as the benchmark, revealing their true performance edge or shortfall in percentage terms that any investor can understand.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            M² builds upon the Sharpe ratio's foundation but transforms it into a more intuitive percentage return measure. Let's explore its mathematical formulation:
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Core Formula</Typography>
          <Typography paragraph>
            The Modigliani Risk-Adjusted Performance (M²) is defined as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>M² Formula</strong></Typography>
            <Equation math="M^2 = S_p \times \sigma_m + \overline{R_f}" />
            <Typography variant="body2">
              where S_p is the portfolio's Sharpe ratio, σ_m is the standard deviation of the market (benchmark), and R̄_f is the average risk-free rate.
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Derivation from the Sharpe Ratio</Typography>
          <Typography paragraph>
            The Sharpe ratio for a portfolio (S_p) is calculated as:
          </Typography>

          <Equation math="S_p = \frac{\overline{R_p} - \overline{R_f}}{\sigma_p}" />

          <Typography paragraph>
            Where:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <InlineMath math="\overline{R_p}" /> is the average return of the portfolio
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="\overline{R_f}" /> is the average risk-free rate
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="\sigma_p" /> is the standard deviation of portfolio returns (volatility)
              </Typography>
            </li>
          </ul>

          <Typography paragraph>
            By substituting and rearranging, M² can also be expressed as:
          </Typography>

          <Equation math="M^2 = \overline{R_p^*} = (\overline{R_p} - \overline{R_f}) \times \frac{\sigma_m}{\sigma_p} + \overline{R_f}" />

          <Typography paragraph>
            This formulation directly shows how M² represents the return of a risk-adjusted portfolio, where:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <InlineMath math="\overline{R_p^*}" /> is the return of the portfolio adjusted to the market's volatility level
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="(\overline{R_p} - \overline{R_f}) \times \frac{\sigma_m}{\sigma_p}" /> is the excess return adjusted for relative volatility
              </Typography>
            </li>
          </ul>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Alternative Expression: M² Alpha</Typography>
          <Typography paragraph>
            A related measure, sometimes called M² Alpha or RAPA (Risk-Adjusted Performance Alpha), represents just the risk-adjusted excess return:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>M² Alpha Formula</strong></Typography>
            <Equation math="M^2_{\alpha} = S_p \times \sigma_m = (\overline{R_p} - \overline{R_f}) \times \frac{\sigma_m}{\sigma_p}" />
            <Typography variant="body2">
              This represents the risk-adjusted excess return above the risk-free rate.
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Practical Interpretation</Typography>
          <Typography paragraph>
            The key insight is that M² measures the return that would be achieved if the portfolio were adjusted (through leverage or de-leveraging) to have the same volatility as the benchmark. This allows for direct comparison with the benchmark return:
          </Typography>

          <Typography paragraph>
            <strong>If M² {'>'}R_m:</strong> The portfolio outperformed the benchmark on a risk-adjusted basis
          </Typography>

          <Typography paragraph>
            <strong>If M² {'<'}R_m:</strong> The portfolio underperformed the benchmark on a risk-adjusted basis
          </Typography>

          <Typography paragraph>
            <strong>If M² = R_m:</strong> The portfolio performed exactly as well as the benchmark on a risk-adjusted basis
          </Typography>

          <Typography paragraph>
            The difference (M² - R_m) represents the portfolio's risk-adjusted excess return compared to the benchmark.
          </Typography>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Portfolio Analysis</Typography>
          <Typography paragraph>
            Our implementation of the Modigliani Risk-Adjusted Performance involves the following key steps:
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                <strong>Data Preparation:</strong> Collect historical return data for the portfolio, benchmark, and risk-free rate over the same time period.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Calculate Return Statistics:</strong> Compute the average returns and standard deviations for the portfolio and benchmark.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Calculate the Sharpe Ratio:</strong> Determine the portfolio's Sharpe ratio by dividing excess return by portfolio volatility.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Apply the M² Formula:</strong> Multiply the Sharpe ratio by the benchmark's standard deviation and add the risk-free rate.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Interpretation:</strong> Compare the resulting M² value directly with the benchmark's average return to determine risk-adjusted outperformance or underperformance.
              </Typography>
            </li>
          </ol>
          <Typography paragraph>
            In portfolio optimization applications, M² can be used in several ways:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Performance Evaluation:</strong> Comparing the risk-adjusted performance of different portfolios or strategies against a common benchmark.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Manager Selection:</strong> Evaluating different portfolio managers' risk-adjusted performance in percentage terms that are intuitive to clients.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Portfolio Optimization:</strong> While not typically used as a direct optimization objective, M² can be used to evaluate outcomes from other optimization methods.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Management:</strong> Understanding how changes in portfolio composition affect risk-adjusted returns relative to a benchmark.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's calculate the M² measure for two different investment portfolios and compare their risk-adjusted performance against a market benchmark.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: Historical Data</Typography>
          <Typography paragraph>
            Assume we have the following annual performance data:
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Typography variant="body2">
              <strong>Risk-free rate:</strong> 3% per annum
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>Market benchmark:</strong> Average return: 10%, Standard deviation (volatility): 15%
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>Portfolio A:</strong> Average return: 12%, Standard deviation: 20%
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>Portfolio B:</strong> Average return: 8%, Standard deviation: 10%
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Calculate Sharpe Ratios</Typography>
          <Typography paragraph>
            First, we calculate the Sharpe ratio for each portfolio:
          </Typography>
          <Grid container spacing={3} sx={{ mb: 2 }}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom><strong>Portfolio A</strong></Typography>
              <Typography paragraph>
                S_A = (12% - 3%) / 20% = 9% / 20% = 0.45
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom><strong>Portfolio B</strong></Typography>
              <Typography paragraph>
                S_B = (8% - 3%) / 10% = 5% / 10% = 0.50
              </Typography>
            </Grid>
          </Grid>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Calculate M² Values</Typography>
          <Typography paragraph>
            Using the formula M² = S × σ_m + R_f:
          </Typography>
          <Grid container spacing={3} sx={{ mb: 2 }}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom><strong>Portfolio A</strong></Typography>
              <Typography paragraph>
                M²_A = 0.45 × 15% + 3% = 6.75% + 3% = <strong>9.75%</strong>
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom><strong>Portfolio B</strong></Typography>
              <Typography paragraph>
                M²_B = 0.50 × 15% + 3% = 7.5% + 3% = <strong>10.5%</strong>
              </Typography>
            </Grid>
          </Grid>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 4: Compare with Benchmark</Typography>
          <Typography paragraph>
            Now we can directly compare the M² values with the market benchmark return of 10%:
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Typography variant="body2">
              <strong>Portfolio A:</strong> M²_A = 9.75% (underperforms benchmark by 0.25%)
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>Portfolio B:</strong> M²_B = 10.5% (outperforms benchmark by 0.5%)
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 5: Interpretation</Typography>
          <Typography paragraph>
            This example reveals some important insights:
          </Typography>
          <Typography paragraph>
            <strong>Portfolio A</strong> has a higher raw return (12%) than both Portfolio B (8%) and the market (10%). However, when adjusted for risk, its performance (M² = 9.75%) actually falls slightly below the market's return of 10%. This suggests that Portfolio A's higher returns don't fully compensate for its higher volatility.
          </Typography>
          <Typography paragraph>
            <strong>Portfolio B</strong> has a lower raw return (8%) than both Portfolio A (12%) and the market (10%). Yet, when adjusted for risk, it demonstrates the best performance (M² = 10.5%), outperforming the market by 0.5%. This indicates that Portfolio B achieves its returns with remarkably low risk, making it the most efficient portfolio on a risk-adjusted basis.
          </Typography>
          <Typography paragraph>
            This is precisely the kind of insight that makes M² valuable—it transforms abstract risk-adjusted comparisons into concrete percentage returns that can be directly compared with benchmark performance, revealing which investment approach truly adds value when accounting for risk.
          </Typography>
        </Paper>

        {/* Practical Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Performance Reporting</Typography>
          <Typography paragraph>
            M² excels in client reporting contexts where portfolio managers need to explain risk-adjusted performance to non-technical audiences. The percentage-based format makes it immediately comprehensible to investors without requiring technical knowledge of statistics or finance theory.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Fund Selection</Typography>
          <Typography paragraph>
            When comparing multiple funds, M² allows investors to see which would provide the best return at a standardized risk level. This facilitates better-informed investment decisions by separating manager skill from risk-taking.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Risk Budgeting</Typography>
          <Typography paragraph>
            In institutional settings, M² helps allocate risk budgets by identifying which strategies deliver the most return per unit of risk. This is particularly valuable when working with a diverse range of investment approaches across multiple asset classes.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Strategy Evaluation</Typography>
          <Typography paragraph>
            For systematic trading strategies, M² provides a framework for comparing performance across various market environments. It helps answer whether a strategy's returns justify its risk profile compared to a passive benchmark approach.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Asset Allocation</Typography>
          <Typography paragraph>
            At the portfolio construction level, M² can inform allocation decisions by highlighting which asset classes or factors deliver superior risk-adjusted returns relative to broad market exposure. This helps build more efficient investment portfolios.
          </Typography>
        </Paper>

        {/* Advantages and Limitations */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Advantages and Limitations</Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="primary">
                  Advantages
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Intuitive Interpretation:</strong> Expresses risk-adjusted performance in percentage returns rather than an abstract ratio, making it immediately meaningful to all investors.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Direct Benchmark Comparison:</strong> Allows for straightforward comparison against benchmark returns, as both are expressed in the same units and adjusted to the same risk level.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Client Communication:</strong> Simplifies the explanation of risk-adjusted performance to clients without requiring technical knowledge of finance or statistics.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Negative Returns:</strong> Unlike the Sharpe ratio, M² remains meaningful and easy to interpret even when dealing with negative returns or Sharpe ratios.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Customizable Benchmark:</strong> Can use any relevant benchmark, not just the market, allowing for appropriate comparisons within specific investment mandates or strategies.
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
                      <strong>Total Volatility Focus:</strong> Like the Sharpe ratio it's derived from, M² treats all volatility equally, without distinguishing between upside and downside risk.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Normal Distribution Assumption:</strong> Implicitly assumes returns are normally distributed, which often doesn't hold for many investment strategies, particularly those with options or alternative assets.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Benchmark Dependency:</strong> Results are highly dependent on the chosen benchmark, potentially leading to different conclusions when different benchmarks are used.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Historical Data Limitations:</strong> As with all backward-looking metrics, past performance data may not reliably predict future results.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Leverage Practicality:</strong> The theoretical leveraging or de-leveraging implied in the calculation may not be practically implementable due to constraints, costs, or leverage limitations.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Time Period Sensitivity:</strong> Results can vary significantly depending on the time period chosen, potentially leading to period-selection bias.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Academic References</Typography>
          <ul>
            <li>
              <Typography paragraph>
                Sharpe, W. F. (1966). "Mutual Fund Performance". Journal of Business. 39(S1): 119–138. doi:10.1086/294846.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Sharpe, William F. (1994). "The Sharpe Ratio". Journal of Portfolio Management. 1994(Fall): 49–58. doi:10.3905/jpm.1994.409501. S2CID 55394403.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Modigliani, Franco (1997). "Risk-Adjusted Performance". Journal of Portfolio Management. 1997(Winter): 45–54. doi:10.3905/jpm.23.2.45. S2CID 154490980.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Modigliani, Leah (1997). "Yes, You Can Eat Risk-Adjusted Returns". Morgan Stanley U.S. Investment Research. 1997(March 17, 1997): 1–4.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Bacon, Carl (2013). "Practical Risk-Adjusted Performance Measurement". Wiley Finance.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Le Sourd, V. (2007). "Performance Measurement for Traditional Investment". EDHEC Risk and Asset Management Research Centre.
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
                  Sharpe Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  The classic risk‐adjusted return metric that divides excess portfolio return by total volatility.
                </Typography>
                <Link href="/education/sharpe-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Information Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A performance metric that evaluates active return per unit of risk relative to a benchmark index.
                </Typography>
                <Link href="/education/information-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Treynor Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A portfolio performance metric that measures returns earned in excess of the risk-free rate per unit of market risk (beta).
                </Typography>
                <Link href="/education/treynor-ratio" passHref>
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

export default ModiglianiRiskAdjustedPage; 