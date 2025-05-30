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

const UpsidePotentialRatioPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Upside Potential Ratio for Indian Portfolios | QuantPort India Docs</title>
        <meta
          name="description"
          content="Optimize Indian stock portfolios using the Upside Potential Ratio. Evaluate NSE/BSE investments by balancing upside opportunity with downside protection in volatile markets."
        />
        <meta property="og:title" content="Upside Potential Ratio for Indian Portfolios | QuantPort India Docs" />
        <meta property="og:description" content="Optimize Indian stock portfolios using the Upside Potential Ratio. Evaluate NSE/BSE investments by balancing upside opportunity with downside protection in volatile markets." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/upside-potential-ratio" />
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
            Upside Potential Ratio
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A performance metric that evaluates upside potential relative to downside risk, focusing on beneficial asymmetry
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            <strong>Upside Potential Ratio</strong> is a sophisticated performance metric that evaluates investment returns by comparing upside potential to downside risk. Unlike traditional metrics such as the Sharpe ratio that treat all volatility equally, the Upside Potential Ratio specifically distinguishes between favorable and unfavorable deviations from a minimum acceptable return.
          </Typography>
          <Typography paragraph>
            Introduced by Frank Sortino and others as an extension of the Sortino ratio, this metric addresses a fundamental reality of investor psychology: investors generally prefer upside volatility (gains) while seeking to minimize downside volatility (losses). By focusing on this asymmetry, the Upside Potential Ratio provides a more nuanced evaluation of investment performance that aligns with investors' actual preferences and risk perceptions.
          </Typography>
          <Typography paragraph>
            This ratio is particularly valuable for evaluating investments with non-normal return distributions, such as those involving options, alternative investments, or strategies that deliberately seek to create asymmetric return profiles. It helps identify investments that offer substantial upside opportunity relative to their downside exposure—a key consideration for investors seeking growth while maintaining risk discipline.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Imagine you're evaluating two different hiking trails to a mountain peak. Both trails have the same average elevation gain, but they differ in an important way:
          </Typography>
          <Typography paragraph>
            <strong>Trail A</strong> has consistent, moderate inclines and declines throughout the journey.
          </Typography>
          <Typography paragraph>
            <strong>Trail B</strong> has mostly gentle declines or flat sections, but occasionally features steep ascents that quickly gain elevation.
          </Typography>
          <Typography paragraph>
            Traditional metrics like the Sharpe ratio would view these trails as roughly equivalent since they focus on average gain relative to overall variability. However, many hikers would prefer Trail B—they'd rather have occasional steep climbs (upside volatility) with mostly easier paths (limited downside) than constant moderate effort.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Financial analogy:</strong> In investing terms, the Upside Potential Ratio helps identify investments that are like Trail B—those that limit losses while maintaining strong potential for gains. It's like measuring how much "oxygen-rich scenic viewpoints" (upside potential) you get relative to the "exhausting downhill scrambles" (downside risk) in your investment journey.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            The Upside Potential Ratio builds upon partial moments, which separate returns into upside and downside components relative to a target return.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Core Formula</Typography>
          <Typography paragraph>
            The Upside Potential Ratio is defined as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Upside Potential Ratio Formula</strong></Typography>
            <Equation math="UPR = \frac{\text{Upside Potential}}{\text{Downside Risk}}" />
            <Typography variant="body2">
              where Upside Potential measures the expected gains above the threshold, and Downside Risk quantifies the expected losses below the threshold.
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Formal Definition</Typography>
          <Typography paragraph>
            More precisely, the Upside Potential Ratio is calculated as:
          </Typography>

          <Equation math="UPR_{\tau} = \frac{\text{UPM}_1(R, \tau)}{\sqrt{\text{LPM}_2(R, \tau)}}" />

          <Typography paragraph>
            Where:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <InlineMath math="\tau" /> is the minimum acceptable return (threshold or target return)
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="\text{UPM}_1(R, \tau)" /> is the first-degree Upper Partial Moment, measuring the average upside deviation
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="\text{LPM}_2(R, \tau)" /> is the second-degree Lower Partial Moment, measuring the variance of downside deviations
              </Typography>
            </li>
          </ul>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Partial Moments Calculation</Typography>
          <Typography paragraph>
            The Upper and Lower Partial Moments are calculated as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Upper Partial Moment (First-degree)</strong></Typography>
            <Equation math="\text{UPM}_1(R, \tau) = \frac{1}{n} \sum_{i=1}^{n} \max(R_i - \tau, 0)" />
            <Typography variant="body2">
              This represents the average of all returns above the threshold <InlineMath math="\tau" />.
            </Typography>
          </Box>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Lower Partial Moment (Second-degree)</strong></Typography>
            <Equation math="\text{LPM}_2(R, \tau) = \frac{1}{n} \sum_{i=1}^{n} (\max(\tau - R_i, 0))^2" />
            <Typography variant="body2">
              This represents the average squared deviation below the threshold <InlineMath math="\tau" />.
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Relationship to Other Metrics</Typography>
          <Typography paragraph>
            The Upside Potential Ratio relates to the Sortino Ratio as follows:
          </Typography>

          <ul>
            <li>
              <Typography paragraph>
                <strong>Sortino Ratio:</strong> <InlineMath math="\text{Sortino} = \frac{E[R] - \tau}{\sqrt{\text{LPM}_2(R, \tau)}}" />
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Upside Potential Ratio:</strong> <InlineMath math="UPR_{\tau} = \frac{\text{UPM}_1(R, \tau)}{\sqrt{\text{LPM}_2(R, \tau)}}" />
              </Typography>
            </li>
          </ul>

          <Typography paragraph>
            The key difference is that the Sortino Ratio uses the average excess return (which can include returns below the threshold), while the Upside Potential Ratio uses only the positive deviations above the threshold.
          </Typography>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Portfolio Analysis</Typography>
          <Typography paragraph>
            Our implementation of the Upside Potential Ratio involves the following steps:
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                <strong>Define the Threshold Return:</strong> We allow users to specify a minimum acceptable return (MAR) based on their investment goals. Common choices include:
              </Typography>
              <ul>
                <li>
                  <Typography paragraph>
                    Zero (evaluating absolute returns)
                  </Typography>
                </li>
                <li>
                  <Typography paragraph>
                    Risk-free rate (measuring excess returns)
                  </Typography>
                </li>
                <li>
                  <Typography paragraph>
                    Inflation rate (preserving purchasing power)
                  </Typography>
                </li>
                <li>
                  <Typography paragraph>
                    Custom target (specific to investor requirements)
                  </Typography>
                </li>
              </ul>
            </li>
            <li>
              <Typography paragraph>
                <strong>Calculate Partial Moments:</strong> Using historical or simulated return data, we separate returns into those above the threshold (for UPM) and those below (for LPM).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Compute the Ratio:</strong> We calculate the Upside Potential Ratio using the formulas described above.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Portfolio Comparison:</strong> We rank portfolios based on their Upside Potential Ratios, highlighting those with the most favorable asymmetry.
              </Typography>
            </li>
          </ol>
          <Typography paragraph>
            In portfolio optimization, we can use the Upside Potential Ratio as:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>An Objective Function:</strong> Constructing portfolios that maximize the upside potential relative to downside risk.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>A Screening Tool:</strong> Filtering investment options to focus on those with the most attractive risk-reward asymmetry.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>A Complementary Metric:</strong> Using it alongside traditional measures like Sharpe ratio to gain a more comprehensive view of performance.
              </Typography>
            </li>
          </ul>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', mt: 2, textAlign: 'center' }}>
            <Typography variant="subtitle2" gutterBottom>Upside Potential Ratio Visualization (Placeholder)</Typography>
            <Box sx={{ height: '300px', bgcolor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                [Placeholder for visualization comparing return distributions and highlighting upside vs. downside regions]
              </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary">
              The chart illustrates how the Upside Potential Ratio separates returns into upside potential and downside risk relative to a threshold return.
            </Typography>
          </Box>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's calculate and compare the Upside Potential Ratio for two hypothetical investments using a minimum acceptable return of 3% (which might represent inflation or a risk-free rate).
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: Historical Returns Data</Typography>
          <Typography paragraph>
            Suppose we have the following annual returns for two investments over a 10-year period:
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Typography variant="body2">
              <strong>Investment A:</strong> 8%, -2%, 15%, 6%, -5%, 10%, 12%, -3%, 7%, 9%
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>Investment B:</strong> 5%, 4%, 6%, -1%, 3%, 8%, 4%, 2%, 5%, 6%
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Calculate Upper and Lower Partial Moments</Typography>
          <Typography paragraph>
            First, we identify returns above and below our threshold of 3%:
          </Typography>
          <Grid container spacing={3} sx={{ mb: 2 }}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom><strong>Investment A</strong></Typography>
              <Typography paragraph>
                <strong>Returns above 3%:</strong> 8%, 15%, 6%, 10%, 12%, 7%, 9%
              </Typography>
              <Typography paragraph>
                <strong>Returns below 3%:</strong> -2%, -5%, -3%
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom><strong>Investment B</strong></Typography>
              <Typography paragraph>
                <strong>Returns above 3%:</strong> 5%, 4%, 6%, 8%, 4%, 5%, 6%
              </Typography>
              <Typography paragraph>
                <strong>Returns below 3%:</strong> -1%, 2%
              </Typography>
            </Grid>
          </Grid>

          <Typography paragraph>
            Now we calculate the partial moments:
          </Typography>
          
          <Typography variant="subtitle2" gutterBottom><strong>For Investment A:</strong></Typography>
          <Typography paragraph>
            UPM₁ = (8-3 + 15-3 + 6-3 + 10-3 + 12-3 + 7-3 + 9-3)/10 = (5+12+3+7+9+4+6)/10 = 46/10 = 4.6
          </Typography>
          <Typography paragraph>
            LPM₂ = [(3-(-2))² + (3-(-5))² + (3-(-3))²]/10 = (5² + 8² + 6²)/10 = (25+64+36)/10 = 125/10 = 12.5
          </Typography>
          
          <Typography variant="subtitle2" sx={{ mt: 2 }} gutterBottom><strong>For Investment B:</strong></Typography>
          <Typography paragraph>
            UPM₁ = (5-3 + 4-3 + 6-3 + 8-3 + 4-3 + 5-3 + 6-3)/10 = (2+1+3+5+1+2+3)/10 = 17/10 = 1.7
          </Typography>
          <Typography paragraph>
            LPM₂ = [(3-(-1))² + (3-2)²]/10 = (4² + 1²)/10 = (16+1)/10 = 17/10 = 1.7
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Calculate Upside Potential Ratio</Typography>
          <Typography paragraph>
            Using our formula UPR = UPM₁ / √LPM₂:
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 2 }}>
            <Typography variant="body2">
              <strong>Investment A:</strong> UPR = 4.6 / √12.5 = 4.6 / 3.54 ≈ 1.30
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>Investment B:</strong> UPR = 1.7 / √1.7 = 1.7 / 1.30 ≈ 1.31
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 4: Interpretation</Typography>
          <Typography paragraph>
            Despite Investment A having a higher average return (5.7% vs. 4.2% for Investment B), both investments have nearly identical Upside Potential Ratios. This indicates that relative to their respective downside risks, they offer similar upside potential.
          </Typography>
          <Typography paragraph>
            Investment B has lower absolute upside potential (1.7 vs. 4.6) but also much lower downside risk (1.7 vs. 12.5), making it potentially more attractive for risk-averse investors. Investment A offers higher potential returns but with correspondingly higher risk.
          </Typography>
          <Typography paragraph>
            This example illustrates how the Upside Potential Ratio can reveal nuances in the risk-return relationship that aren't apparent from average returns alone. Depending on an investor's risk tolerance, they might prefer Investment B's more consistent performance despite its lower average return.
          </Typography>
        </Paper>

        {/* Practical Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Strategy Evaluation</Typography>
          <Typography paragraph>
            The Upside Potential Ratio is particularly useful for evaluating investment strategies designed to capture upside while limiting downside, such as option-based strategies, structured products, and tactical asset allocation approaches. It helps identify strategies that deliver on their promise of asymmetric returns.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Manager Selection</Typography>
          <Typography paragraph>
            When comparing fund managers, especially those employing active strategies, the Upside Potential Ratio helps identify managers who effectively capture bull markets while providing protection during bear markets—a key skill that may not be evident from Sharpe or even Sortino ratios alone.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Portfolio Construction</Typography>
          <Typography paragraph>
            By optimizing for Upside Potential Ratio rather than traditional risk-adjusted returns, investors can construct portfolios that align better with their asymmetric preferences for gains versus losses, potentially improving perceived satisfaction with investment outcomes.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Risk Management</Typography>
          <Typography paragraph>
            Using the Upside Potential Ratio in risk management helps focus attention on limiting downside deviations that matter to investors, rather than treating all volatility as equally undesirable. This aligns risk management practices more closely with investor psychology and goals.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Behavioral Finance</Typography>
          <Typography paragraph>
            The ratio acknowledges the behavioral reality that investors feel the pain of losses more acutely than the pleasure of equivalent gains (loss aversion). By explicitly measuring upside separately from downside, it provides a metric that better reflects how investors actually experience and evaluate their investments.
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
                      <strong>Psychological Alignment:</strong> Better reflects investor preferences by distinguishing between favorable and unfavorable volatility, recognizing that investors desire upside potential while seeking to minimize downside risk.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Adaptable Threshold:</strong> The minimum acceptable return can be customized to match specific investment goals, allowing for personalized performance evaluation.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Non-Normal Distributions:</strong> More appropriate than traditional metrics for evaluating investments with skewed return distributions, such as those involving options, alternative investments, or asymmetric strategies.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Focus on Upside Capture:</strong> Explicitly rewards strategies that maximize gains above the threshold, not just those that minimize overall volatility.
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
                      <strong>Threshold Sensitivity:</strong> Results can be highly dependent on the chosen minimum acceptable return, requiring careful consideration when selecting this parameter.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Interpretation Complexity:</strong> Less intuitive than simpler metrics like the Sharpe ratio, potentially making it more difficult to explain to investors without a technical background.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Data Requirements:</strong> Requires sufficient historical data to reliably estimate upside potential and downside risk, which may not always be available for newer investment strategies.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Limited Standardization:</strong> Not as widely used or standardized as traditional metrics, making cross-industry comparisons more challenging.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Time Insensitivity:</strong> Like many risk-adjusted return measures, it does not account for the timing of returns or drawdowns, which can be important for investors with specific time horizons.
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
                Sortino, F. A., & Price, L. N. (1994). "Performance Measurement in a Downside Risk Framework." Journal of Investing, 3(3), 59-64.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Sortino, F., van der Meer, R., & Plantinga, A. (1999). "The Dutch Triangle: A Framework to Measure Upside Potential Relative to Downside Risk." Journal of Portfolio Management, 26(1), 50-58.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Fishburn, P. C. (1977). "Mean-Risk Analysis with Risk Associated with Below-Target Returns." American Economic Review, 67(2), 116-126.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Kahneman, D., & Tversky, A. (1979). "Prospect Theory: An Analysis of Decision under Risk." Econometrica, 47(2), 263-291.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Plantinga, A., & de Groot, S. (2001). "Risk-Adjusted Performance Measures and Implied Risk-Attitudes." Journal of Performance Measurement, 6(2), 9-22.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Bawa, V. S. (1975). "Optimal Rules for Ordering Uncertain Prospects." Journal of Financial Economics, 2(1), 95-121.
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
          Sortino Ratio
        </Typography>
        <Typography variant="body2" paragraph>
          A downside‐risk focused performance measure that divides excess return by the downside deviation below a target.
        </Typography>
        <Link href="/docs/sortino-ratio" passHref>
          <Button variant="contained" color="primary">
            Learn More
          </Button>
        </Link>
      </Box>
    </Grid>

    <Grid item xs={12} sm={6} md={4}>
      <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
        <Typography variant="h6" gutterBottom>
          Omega Ratio
        </Typography>
        <Typography variant="body2" paragraph>
          A threshold‐based performance metric that compares the probability‐weighted gains to losses relative to a benchmark.
        </Typography>
        <Link href="/docs/omega-ratio" passHref>
          <Button variant="contained" color="primary">
            Learn More
          </Button>
        </Link>
      </Box>
    </Grid>

    <Grid item xs={12} sm={6} md={4}>
      <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
        <Typography variant="h6" gutterBottom>
          Sharpe Ratio
        </Typography>
        <Typography variant="body2" paragraph>
          The classic risk‐adjusted return metric that divides excess portfolio return by total volatility.
        </Typography>
        <Link href="/docs/sharpe-ratio" passHref>
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

export default UpsidePotentialRatioPage; 