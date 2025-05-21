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

const EntropyPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Entropy in Portfolio Returns | Portfolio Optimization</title>
        <meta
          name="description"
          content="Learn about Entropy as a measure of uncertainty or randomness in portfolio returns, including the Freedman-Diaconis rule for bin width calculation."
        />
      </Head>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Education</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* Title */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Entropy in Portfolio Returns
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Quantifying uncertainty and randomness in financial markets
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            <strong>Entropy</strong> is a fundamental concept borrowed from information theory and thermodynamics that measures the level of uncertainty, 
            disorder, or randomness in a system. In portfolio analysis, entropy quantifies the unpredictability of returns, providing a 
            non-parametric measure of risk that doesn't rely on assumptions about normal distributions. 
          </Typography>
          <Typography paragraph>
            Unlike variance which only captures dispersion around the mean, entropy considers the entire probability 
            distribution of returns, making it particularly valuable when dealing with complex, non-normal market behaviors 
            such as fat tails, skewness, and multimodality.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Think of entropy as measuring how "surprising" or "unpredictable" future portfolio returns might be. 
            A high-entropy portfolio behaves like a highly unpredictable system—returns could come from almost 
            anywhere in the distribution with similar probabilities. This makes planning difficult since future 
            outcomes are highly uncertain.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Weather analogy:</strong> Consider two cities. In City A, it rains 50% of days, distributed evenly throughout the year. 
              In City B, it rains exactly 50% of days too, but only during monsoon season when it rains every day. City A has high entropy 
              (any given day might be rainy or sunny with equal probability), while City B has lower entropy (the weather is highly predictable 
              based on the season). Even though both have the same average rainfall, they have different entropy profiles.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            Shannon's entropy measures the expected information content or surprise in a random variable. For portfolio returns
            discretized into bins, the entropy is defined as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Shannon Entropy Formula</strong></Typography>
            <Equation math="H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)" />
            <Typography variant="body2">
              where <InlineMath math="p_i" /> is the probability of returns falling into bin <InlineMath math="i" />, and 
              <InlineMath math="n" /> is the number of bins. By convention, <InlineMath math="0 \log_2(0) = 0" />.
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom>The Freedman-Diaconis Rule for Bin Width</Typography>
          <Typography paragraph>
            To calculate entropy from empirical return data, we must first discretize continuous returns into bins.
            The choice of bin width significantly impacts entropy estimation—too few bins oversimplifies the distribution,
            while too many creates noise from sparse sampling.
          </Typography>

          <Typography paragraph>
            The <strong>Freedman-Diaconis rule</strong> provides an optimal bin width that balances these concerns:
          </Typography>

          <Equation math="\text{Bin width} = 2 \cdot \frac{IQR(X)}{\sqrt[3]{n}}" />

          <Typography paragraph>
            where <InlineMath math="IQR(X)" /> is the interquartile range (Q3 - Q1) of the data and <InlineMath math="n" /> is the number 
            of observations. This method is robust to outliers because it uses the IQR rather than the standard deviation.
          </Typography>

          <Typography paragraph>
            The number of bins is then calculated as:
          </Typography>

          <Equation math="\text{Number of bins} = \frac{\max(X) - \min(X)}{\text{Bin width}}" />

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Normalized Entropy</Typography>
          <Typography paragraph>
            For easier interpretation, we often normalize entropy to a [0,1] scale by dividing by the maximum possible entropy (uniform distribution):
          </Typography>

          <Equation math="H_{norm}(X) = \frac{H(X)}{\log_2(n)}" />

          <Typography paragraph>
            A normalized entropy of 1 represents maximum uncertainty (uniform distribution), while values closer to 0 indicate more concentrated,
            predictable return patterns.
          </Typography>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Our Service</Typography>
          <Typography paragraph>
            Our portfolio analyzer calculates entropy from historical returns through the following process:
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                <strong>Bin Width Calculation:</strong> We apply the Freedman-Diaconis rule to determine optimal bin width based on the interquartile range and sample size.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Histogram Construction:</strong> Returns are discretized into bins, and frequencies are converted to probabilities.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Entropy Calculation:</strong> Shannon's entropy formula is applied to the probability distribution.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Normalization:</strong> The raw entropy value is normalized by dividing by <InlineMath math="\log_2(n)" /> to yield a value between 0 and 1.
              </Typography>
            </li>
          </ol>
          <Typography paragraph>
            This implementation provides a robust measure of return uncertainty that complements traditional risk metrics like variance or Value-at-Risk.
          </Typography>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Consider 1000 daily returns from a portfolio with values ranging from -3% to +3%. If the IQR is 0.8%:
          </Typography>
          <Typography component="div" paragraph>
            <strong>Step 1:</strong> Calculate bin width using the Freedman-Diaconis rule:
            <Equation math="\text{Bin width} = 2 \cdot \frac{0.008}{\sqrt[3]{1000}} \approx 0.0016 \text{ or } 0.16\%" />
          </Typography>
          <Typography component="div" paragraph>
            <strong>Step 2:</strong> Determine the number of bins:
            <Equation math="\text{Number of bins} = \frac{0.03 - (-0.03)}{0.0016} \approx 37.5 \approx 38 \text{ bins}" />
          </Typography>
          <Typography component="div" paragraph>
            <strong>Step 3:</strong> Construct histogram and calculate probabilities for each bin.
          </Typography>
          <Typography component="div" paragraph>
            <strong>Step 4:</strong> Calculate entropy using Shannon's formula. If the resulting value is 4.8 bits:
            <Equation math="H_{norm} = \frac{4.8}{\log_2(38)} \approx \frac{4.8}{5.25} \approx 0.91" />
          </Typography>
          <Typography paragraph>
            This normalized entropy of 0.91 indicates the return distribution is quite uncertain (close to uniform), suggesting high unpredictability
            in this portfolio's performance.
          </Typography>
        </Paper>

        {/* Why Entropy Matters */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Why Entropy Matters in Portfolio Management</Typography>
          <Typography paragraph>
            Entropy provides several key insights that traditional risk measures may miss:
          </Typography>
          
          <ul>
            <li>
              <Typography paragraph>
                <strong>Beyond Variance:</strong> While variance only captures dispersion around the mean, entropy describes the shape and 
                concentration of the entire distribution. Two portfolios with identical variance can have dramatically different entropy values.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Non-Parametric Nature:</strong> Entropy doesn't assume returns follow any particular distribution, making it valuable for 
                markets with fat tails, skewness, and other non-normal characteristics.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Diversification Quality:</strong> Entropy can indicate true diversification benefits better than correlation alone. A 
                well-diversified portfolio typically has lower entropy than the sum of its parts.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Market Regime Detection:</strong> Sudden changes in portfolio entropy can signal shifts in market regimes before they become 
                apparent in other metrics, potentially providing early warning of changing conditions.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Preferences:</strong> Some investors may prefer low-entropy portfolios (more predictable outcomes) even at the cost of 
                slightly lower expected returns, particularly for specific goals like retirement planning.
              </Typography>
            </li>
          </ul>
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
                      <strong>Distribution-agnostic:</strong> Works without assuming normality in returns, capturing information about fat tails and asymmetry.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Comprehensive risk view:</strong> Considers the entire probability distribution rather than just dispersion around the mean.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Information content:</strong> Measures the actual information content or surprise in returns, which is fundamental to pricing efficiency.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Regime detection:</strong> Can identify shifts in market conditions through changes in the return distribution's structure.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Complementary metric:</strong> Provides additional insight when used alongside traditional risk measures like VaR, standard deviation, and drawdown.
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
                      <strong>Bin sensitivity:</strong> Entropy calculations depend on binning choices, which can introduce methodological bias if not carefully implemented.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Data requirements:</strong> Requires substantial historical data to reliably estimate the probability distribution.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Interpretability:</strong> Less intuitive for practitioners compared to traditional risk measures like standard deviation or maximum drawdown.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Time-insensitivity:</strong> Basic entropy doesn't account for the temporal ordering of returns, missing serial dependencies and volatility clustering.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Lack of directional information:</strong> Doesn't distinguish between upside and downside uncertainty, which have different implications for investors.
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
              <Typography paragraph><strong>Shannon, C. E. (1948)</strong>. "A Mathematical Theory of Communication." <em>Bell System Technical Journal</em>, 27(3), 379-423.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Dionisio, A., Menezes, R., & Mendes, D. A. (2006)</strong>. "An econophysics approach to analyse uncertainty in financial markets: an application to the Portuguese stock market." <em>The European Physical Journal B</em>, 50(1), 161-164.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Ormos, M., & Zibriczky, D. (2014)</strong>. "Entropy-based financial asset pricing." <em>PloS one</em>, 9(12), e115742.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Freedman, D., & Diaconis, P. (1981)</strong>. "On the histogram as a density estimator: L2 theory." <em>Probability Theory and Related Fields</em>, 57(4), 453-476.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Zhou, R., Cai, R., & Tong, G. (2013)</strong>. "Applications of entropy in finance: A review." <em>Entropy</em>, 15(11), 4909-4931.</Typography>
            </li>
          </ul>
        </Paper>

        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Related Topics</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Value-at-Risk (VaR)</Typography>
                <Typography variant="body2" paragraph>Another approach to quantifying tail risk and potential losses in a portfolio.</Typography>
                <Link href="/docs/value-at-risk" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Skewness</Typography>
                <Typography variant="body2" paragraph>Distribution asymmetry metric that complements entropy in understanding return patterns.</Typography>
                <Link href="/docs/skewness" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Kurtosis</Typography>
                <Typography variant="body2" paragraph>Measures the "tailedness" of a probability distribution of returns.</Typography>
                <Link href="/docs/kurtosis" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </>
  );
};

// Static generation hook (Next.js)
export const getStaticProps = async () => {
  return { props: {} };
};

export default EntropyPage; 