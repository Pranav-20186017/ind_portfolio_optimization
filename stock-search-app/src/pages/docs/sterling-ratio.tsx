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

const SterlingRatioPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Sterling Ratio for Indian Investments | QuantPort India Docs</title>
        <meta
          name="description"
          content="Evaluate Indian equity portfolios using the Sterling Ratio. Assess risk-adjusted returns of NSE/BSE securities by analyzing average drawdowns with this specialized metric for Indian markets."
        />
        <meta property="og:title" content="Sterling Ratio for Indian Investments | QuantPort India Docs" />
        <meta property="og:description" content="Evaluate Indian equity portfolios using the Sterling Ratio. Assess risk-adjusted returns of NSE/BSE securities by analyzing average drawdowns with this specialized metric for Indian markets." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/sterling-ratio" />
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
            Sterling Ratio
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A risk-adjusted return metric similar to Calmar but using average annual drawdown minus 10% in the denominator
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            <strong>Sterling Ratio</strong> is a risk-adjusted performance metric that evaluates investment returns in relation to downside risk. Developed by Deane Sterling Jones in the 1980s, this ratio was initially designed for evaluating commodity trading advisors (CTAs) and hedge funds but has since been adopted more broadly across various investment strategies.
          </Typography>
          <Typography paragraph>
            The Sterling Ratio modifies the well-known Calmar Ratio by using the average annual maximum drawdown minus an arbitrary 10% buffer in its denominator, rather than simply using the maximum drawdown. This modification aims to account for the fact that drawdowns are an inevitable part of investing and to prevent exceptional but rare drawdown events from disproportionately affecting the ratio.
          </Typography>
          <Typography paragraph>
            By incorporating this adjustment, the Sterling Ratio provides investors with a nuanced view of risk-adjusted performance that may be more representative of typical risk levels encountered in an investment strategy over time, rather than being overly influenced by a single extreme event.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Imagine you're comparing two mountain climbers on their ability to ascend various peaks safely and efficiently. The height they reach represents returns, while the drops they encounter along the way represent drawdowns.
          </Typography>
          <Typography paragraph>
            The Calmar Ratio would look at each climber's total height gained divided by their single worst fall. But a climber might have one unusually bad fall due to a rare weather event, while otherwise maintaining excellent stability.
          </Typography>
          <Typography paragraph>
            The Sterling Ratio takes a different approach: it looks at the average of their significant falls over several climbing seasons and subtracts a "normal fall allowance" of 10%. This provides a more balanced view of each climber's risk-adjusted performance by not allowing a single extreme event to dominate the assessment.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Financial analogy:</strong> If the Sharpe Ratio is like measuring how far you can drive per gallon of gas (returns per unit of overall volatility) and the Calmar Ratio is like measuring how far you can drive before experiencing a complete breakdown (returns per maximum drawdown), the Sterling Ratio measures how far you can drive accounting for regular maintenance issues minus expected wear and tear (returns per average of annual maximum drawdowns minus 10%).
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            The Sterling Ratio is formally defined as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Sterling Ratio Formula</strong></Typography>
            <Equation math="\text{Sterling Ratio} = \frac{R}{\text{AAD} - 10\%}" />
            <Typography variant="body2">
              where <InlineMath math="R" /> is the average annual rate of return, and <InlineMath math="\text{AAD}" /> is the average of annual maximum drawdowns over a specific time period (typically 3-5 years).
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Drawdown Calculation</Typography>
          <Typography paragraph>
            To calculate the average annual maximum drawdown:
          </Typography>

          <ol>
            <li>
              <Typography paragraph>
                For each calendar year in the evaluation period, identify the maximum peak-to-trough decline (maximum drawdown):
              </Typography>
              <Equation math="\text{Annual Maximum Drawdown}_i = \max_{t,s \in \text{Year}_i, t > s} \left( \frac{V_s - V_t}{V_s} \right)" />
              <Typography paragraph>
                where <InlineMath math="V_t" /> is the portfolio value at time t.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Calculate the average of these annual maximum drawdowns:
              </Typography>
              <Equation math="\text{AAD} = \frac{1}{N} \sum_{i=1}^{N} \text{Annual Maximum Drawdown}_i" />
              <Typography paragraph>
                where N is the number of years in the evaluation period.
              </Typography>
            </li>
          </ol>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>The 10% Adjustment</Typography>
          <Typography paragraph>
            The subtraction of 10% in the denominator serves several purposes:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Recognition of normal market fluctuations:</strong> It acknowledges that some level of drawdown is expected in any investment strategy.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Preventing division by zero or negative values:</strong> In cases where the average annual drawdown is less than 10%, the ratio would use a positive value in the denominator.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk normalization:</strong> It establishes a baseline threshold for what constitutes "significant" risk.
              </Typography>
            </li>
          </ul>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Relationship to Other Metrics</Typography>
          <Typography paragraph>
            The Sterling Ratio can be related to other risk metrics as follows:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Calmar Ratio:</strong> <InlineMath math="\text{Calmar Ratio} = \frac{R}{\text{MDD}}" />, where MDD is the maximum drawdown over the entire period. The Sterling Ratio adjusts this by using average annual drawdowns minus 10%.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Sterling-Calmar Ratio:</strong> A variant where <InlineMath math="\text{Sterling-Calmar} = \frac{R}{\text{AAD}}" /> without the 10% adjustment.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Burke Ratio:</strong> A related metric that uses the square root of the sum of squared drawdowns in its denominator.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Portfolio Analysis</Typography>
          <Typography paragraph>
            In our portfolio optimization service, we calculate the Sterling Ratio using the following approach:
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                <strong>Annual Return Calculation:</strong> We compute the compounded annual growth rate (CAGR) of the portfolio over the evaluation period.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Drawdown Identification:</strong> For each calendar year, we identify all drawdowns and determine the maximum drawdown for that year.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Average Calculation:</strong> We calculate the arithmetic mean of these annual maximum drawdowns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Adjustment Application:</strong> We subtract 10% from the average annual maximum drawdown.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Ratio Computation:</strong> We divide the annualized return by the adjusted average drawdown figure.
              </Typography>
            </li>
          </ol>
          <Typography paragraph>
            This implementation allows investors to:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Compare multiple strategies:</strong> The Sterling Ratio provides a standardized metric for comparing different investment approaches.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Evaluate managers:</strong> It helps assess whether investment managers are delivering returns commensurate with the risks they are taking.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Set performance expectations:</strong> The ratio assists in establishing realistic performance targets that account for typical drawdown patterns.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's calculate the Sterling Ratio for a hypothetical investment fund over a 5-year period:
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: Annual Returns</Typography>
          <Typography paragraph>
            Suppose the fund has the following annual returns:
          </Typography>
          <ul>
            <li>Year 1: +15%</li>
            <li>Year 2: +8%</li>
            <li>Year 3: -3%</li>
            <li>Year 4: +20%</li>
            <li>Year 5: +12%</li>
          </ul>
          <Typography paragraph>
            The average annual return is: (15% + 8% - 3% + 20% + 12%) ÷ 5 = 10.4%
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Annual Maximum Drawdowns</Typography>
          <Typography paragraph>
            The maximum drawdown observed in each year:
          </Typography>
          <ul>
            <li>Year 1: 7%</li>
            <li>Year 2: 12%</li>
            <li>Year 3: 18%</li>
            <li>Year 4: 5%</li>
            <li>Year 5: 9%</li>
          </ul>
          <Typography paragraph>
            The average annual maximum drawdown is: (7% + 12% + 18% + 5% + 9%) ÷ 5 = 10.2%
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Apply the Sterling Ratio Formula</Typography>
          <Typography paragraph>
            Sterling Ratio = Average Annual Return ÷ (Average Annual Maximum Drawdown - 10%)
          </Typography>
          <Typography paragraph>
            Sterling Ratio = 10.4% ÷ (10.2% - 10%) = 10.4% ÷ 0.2% = 52
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 4: Interpretation</Typography>
          <Typography paragraph>
            A Sterling Ratio of 52 is quite high, indicating that the fund generates substantial returns relative to its risk-adjusted drawdowns. This high value results partly from the fact that the average annual maximum drawdown (10.2%) is only slightly above the 10% buffer.
          </Typography>
          <Typography paragraph>
            For comparison, let's also calculate the Calmar Ratio using the largest overall drawdown in the entire period, which is 18%:
          </Typography>
          <Typography paragraph>
            Calmar Ratio = 10.4% ÷ 18% = 0.58
          </Typography>
          <Typography paragraph>
            The significant difference between the Sterling Ratio (52) and the Calmar Ratio (0.58) illustrates how the Sterling Ratio's adjustments can dramatically affect the assessment of risk-adjusted performance, especially when a single bad year (Year 3 in this example) contains a drawdown much larger than other years.
          </Typography>
        </Paper>

        {/* Practical Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          <Typography paragraph>
            The Sterling Ratio serves several important functions in investment analysis and portfolio management:
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Manager Selection</Typography>
          <Typography paragraph>
            When evaluating investment managers, particularly in alternative investments like hedge funds or managed futures, the Sterling Ratio helps identify those who deliver consistent returns while managing drawdown risk effectively. It's especially valuable for strategies that experience regular but moderate drawdowns rather than rare but severe ones.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Risk-Adjusted Performance Comparison</Typography>
          <Typography paragraph>
            The Sterling Ratio enables more nuanced comparison between different investment strategies or asset classes by accounting for their typical drawdown patterns. This is particularly useful when comparing strategies with different risk profiles or market exposures.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Trend-Following Strategy Evaluation</Typography>
          <Typography paragraph>
            Trend-following strategies often experience numerous small drawdowns but can perform well over time. The Sterling Ratio is well-suited for evaluating such strategies because it doesn't overly penalize the frequent small drawdowns that are characteristic of these approaches.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Due Diligence Process</Typography>
          <Typography paragraph>
            Institutional investors and fund allocators incorporate the Sterling Ratio into their due diligence processes to ensure that investment strategies are delivering returns commensurate with the risks taken. The ratio's treatment of drawdowns aligns well with how many institutional investors conceptualize risk.
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
                      <strong>Balanced risk assessment:</strong> By using average annual drawdowns rather than the single worst drawdown, the Sterling Ratio provides a more representative picture of typical risk levels.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Reduced volatility in ratio:</strong> The 10% adjustment helps stabilize the ratio by establishing a minimum risk threshold, reducing sensitivity to small changes in drawdown levels.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Realistic risk expectations:</strong> The ratio acknowledges that some level of drawdown is normal and expected in any investment strategy.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Long-term focus:</strong> By considering multiple years of drawdown data, the Sterling Ratio encourages a longer-term investment perspective.
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
                      <strong>Arbitrary adjustment:</strong> The 10% subtraction is somewhat arbitrary and may not be appropriate for all investment strategies or market conditions.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Potential for negative denominator:</strong> If the average annual maximum drawdown is less than 10%, the denominator becomes negative, making the ratio difficult to interpret.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Sensitivity to time period:</strong> The choice of evaluation period can significantly affect the calculated ratio, especially if years with unusual drawdowns are included or excluded.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Simplifies complex risk patterns:</strong> By using an average, the ratio may obscure important information about the frequency and timing of drawdowns.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Limited standardization:</strong> Unlike more widely used metrics like the Sharpe Ratio, there is less consensus about how the Sterling Ratio should be calculated and interpreted.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>

                {/* References */}        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>          <Typography variant="h5" component="h2" gutterBottom>References</Typography>          <ul>            <li>              <Typography paragraph>                Jones, D. S. (1981). "The Sterling Ratios." The Handbook of Stock Index Futures and Options.              </Typography>            </li>            <li>              <Typography paragraph>                Bacon, C. R. (2013). "Practical Risk-Adjusted Performance Measurement." Wiley Finance.              </Typography>            </li>            <li>              <Typography paragraph>                Lhabitant, F. S. (2004). "Hedge Funds: Quantitative Insights." Wiley Finance.              </Typography>            </li>            <li>              <Typography paragraph>                Young, T. W. (1991). "Calmar Ratio: A Smoother Tool." Futures, 20(1).              </Typography>            </li>            <li>              <Typography paragraph>                Schuhmacher, F., & Eling, M. (2011). "Sufficient conditions for expected utility to imply drawdown-based performance rankings." Journal of Banking & Finance, 35(9), 2311-2318.              </Typography>            </li>          </ul>        </Paper>        {/* Related Topics */}        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>          <Typography variant="h5" component="h2" gutterBottom>            Related Topics          </Typography>          <Grid container spacing={2}>            <Grid item xs={12} sm={6} md={4}>              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>                <Typography variant="h6" gutterBottom>                  Calmar Ratio                </Typography>                <Typography variant="body2" paragraph>                  A performance measurement that uses the ratio of average annual compound rate of return to maximum drawdown.                </Typography>                <Link href="/docs/calmar-ratio" passHref>                  <Button variant="contained" color="primary">                    Learn More                  </Button>                </Link>              </Box>            </Grid>            <Grid item xs={12} sm={6} md={4}>              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>                <Typography variant="h6" gutterBottom>                  Maximum Drawdown                </Typography>                <Typography variant="body2" paragraph>                  A measure of the largest peak-to-trough decline in a portfolio's value, representing the worst-case scenario for an investment.                </Typography>                <Link href="/docs/maximum-drawdown" passHref>                  <Button variant="contained" color="primary">                    Learn More                  </Button>                </Link>              </Box>            </Grid>            <Grid item xs={12} sm={6} md={4}>              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>                <Typography variant="h6" gutterBottom>                  Sortino Ratio                </Typography>                <Typography variant="body2" paragraph>                  A modification of the Sharpe ratio that only penalizes returns falling below a specified target or required rate of return.                </Typography>                <Link href="/docs/sortino-ratio" passHref>                  <Button variant="contained" color="primary">                    Learn More                  </Button>                </Link>              </Box>            </Grid>          </Grid>        </Paper>
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

export default SterlingRatioPage; 