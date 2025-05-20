import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

// Reusable Equation component for consistent math rendering
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const SkewnessPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Skewness | Portfolio Optimization</title>
        <meta name="description" content="Learn about Skewness, a measure of asymmetry in return distributions that helps identify strategies with large gains or large losses." />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/education" passHref>
            <Button variant="outlined" color="primary">
              ← Back to Education
            </Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">
              ← Back to Portfolio Optimizer
            </Button>
          </Link>
        </Box>
        
        {/* Title Section */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Skewness
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Understanding asymmetry in return distributions
          </Typography>
        </Box>
        
        {/* What Is Skewness */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            What Is Skewness?
          </Typography>
          <Typography paragraph>
            <strong>Skewness</strong> measures the <strong>asymmetry</strong> of a return distribution:
          </Typography>
          <Box sx={{ pl: 3, mb: 2 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>Right-skewed (positive skew)</strong><br/>
                  Fat right-tail → occasional large <strong>gains</strong>.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Left-skewed (negative skew)</strong><br/>
                  Fat left-tail → occasional large <strong>losses</strong>.
                </Typography>
              </li>
            </ul>
          </Box>
          <Typography paragraph>
            In portfolio analytics skewness helps answer:
          </Typography>
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mb: 2, fontStyle: 'italic', borderLeft: '4px solid #2196f3' }}>
            <Typography>
              "Does this strategy make small profits most of the time and rare blow-ups, or the opposite?"
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Definition */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Mathematical Definition
          </Typography>
          <Typography paragraph>
            For a series of <strong>demeaned</strong> returns <InlineMath math="r_t" /> (<InlineMath math="t=1\dots n" />) with standard deviation <InlineMath math="\sigma" />:
          </Typography>
          <Equation math="\text{Skewness} = \gamma_1 = \frac{1}{n\sigma^3}\sum_{t=1}^{n} r_t^3" />
          
          <Box sx={{ pl: 3, mb: 2 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <InlineMath math="\gamma_1 > 0" /> → right-skew.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <InlineMath math="\gamma_1 < 0" /> → left-skew.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <InlineMath math="\gamma_1 = 0" /> → perfect symmetry (Gaussian).
                </Typography>
              </li>
            </ul>
          </Box>
          <Typography paragraph>
            Because the third moment amplifies tails, even a handful of extreme days can swing skewness dramatically.
          </Typography>
        </Paper>
        
        {/* How Your Backend Calculates Skewness */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            How Our Backend Calculates Skewness
          </Typography>
          <Typography paragraph>
            Inside <strong>srv.py → compute_custom_metrics()</strong> we capture it in one line:
          </Typography>
          <Box 
            component="pre" 
            sx={{ 
              p: 2, 
              bgcolor: '#f5f5f5', 
              borderRadius: 1, 
              overflow: 'auto',
              fontFamily: 'monospace',
              fontSize: '0.875rem'
            }}
          >
{`skewness = port_returns.skew()`}
          </Box>
          
          <Box sx={{ pl: 3, mt: 2 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <code>port_returns</code> are daily <em>simple</em> returns (not log).
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <code>pandas.Series.skew()</code> uses the <strong>Fisher-Pearson</strong> (bias-adjusted) estimator, matching the formula above.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  The scalar is then stored in <code>performance.skewness</code> and surfaced in the results table for every optimisation method (MVO, MinVol, HRP, CLA, …).
                </Typography>
              </li>
            </ul>
          </Box>
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mt: 2, fontStyle: 'italic', borderLeft: '4px solid #2196f3' }}>
            <Typography>
              <strong>Tip:</strong> When you generate the distribution plot (histogram) the fat-tail visual directly matches the sign of this skewness value.
            </Typography>
          </Box>
        </Paper>
        
        {/* Interpreting Skewness in Portfolios */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpreting Skewness in Portfolios
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Skewness</strong></TableCell>
                  <TableCell><strong>Typical Strategy Profile</strong></TableCell>
                  <TableCell><strong>Risk Narrative</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong><InlineMath math="\gamma_1 > 1" /></strong></TableCell>
                  <TableCell>Trend-followers, lottery-like stocks</TableCell>
                  <TableCell>Many small losses, rare big wins</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong><InlineMath math="0 < \gamma_1 < 1" /></strong></TableCell>
                  <TableCell>Broad equity indices</TableCell>
                  <TableCell>Mild right-tail (crash up-days)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong><InlineMath math="-1 < \gamma_1 < 0" /></strong></TableCell>
                  <TableCell>High-yield credit, carry trades</TableCell>
                  <TableCell>Mildly left-tailed (bleed)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong><InlineMath math="\gamma_1 < -1" /></strong></TableCell>
                  <TableCell>Short-vol / options selling</TableCell>
                  <TableCell>Small steady gains, rare large losses</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          <Typography paragraph>
            Combine skewness with <strong>kurtosis</strong> (tail-fatness) for a fuller shape picture.
          </Typography>
        </Paper>
        
        {/* Why It Matters */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Why It Matters
          </Typography>
          <Box sx={{ pl: 3 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>Risk Management</strong> – Left-skewed strategies need extra tail-risk controls (stop-loss, hedges).
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Product Design</strong> – Retail investors often prefer right-skew ("lottery ticket" pay-offs).
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Optimization Constraints</strong> – Advanced portfolio construction (e.g., <strong>Higher-moment MVO</strong>) can set skewness targets.
                </Typography>
              </li>
            </ul>
          </Box>
        </Paper>
        
        {/* Visual Diagnostics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Visual Diagnostics
          </Typography>
          <Typography paragraph>
            Our portfolio optimization app uses histograms with Freedman-Diaconis bins to visualize return distributions, which helps in identifying skewness patterns in your portfolio.
          </Typography>
          <Typography paragraph>
            These histograms provide an intuitive way to see the asymmetry in return distributions and complement the numeric skewness measure.
          </Typography>
        </Paper>
        
        {/* Caveats */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Caveats
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Issue</strong></TableCell>
                  <TableCell><strong>Comment</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Sample Sensitivity</strong></TableCell>
                  <TableCell>Few extreme values can dominate <InlineMath math="\gamma_1" />. Robust checks (bootstrap) are wise.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Non-stationarity</strong></TableCell>
                  <TableCell>Rolling skewness (e.g., 1-year window) reveals regime shifts.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Interpretation vs. Utility</strong></TableCell>
                  <TableCell>Positive skew <strong>isn't automatically good</strong> if gains are tiny; always look at mean & risk metrics together.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Academic References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Academic References
          </Typography>
          <Box sx={{ pl: 3 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>Kraus, A., & Litzenberger, R. H. (1976).</strong> "Skewness Preference and the Valuation of Risk Assets." <em>Journal of Finance</em>, 31(4), 1085-1100.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Harvey, C. R., & Siddique, A. (2000).</strong> "Conditional Skewness in Asset Pricing Tests." <em>Journal of Finance</em>, 55(3), 1263-1295.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Cont, R. (2001).</strong> "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues." <em>Quantitative Finance</em>, 1(2), 223-236.
                </Typography>
              </li>
            </ul>
          </Box>
        </Paper>
        
        {/* Conclusion */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography paragraph sx={{ fontStyle: 'italic' }}>
            By measuring skewness alongside Sharpe, Sortino, and β you gain a <strong>shape dimension</strong> of risk—vital for understanding tail behaviour that variance-based metrics alone can't reveal.
          </Typography>
        </Paper>
        
        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4 }}>
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
                  Standard risk-adjusted return metric that focuses on volatility rather than distributional shape.
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
                  Sortino Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  Risk-adjusted measure that focuses on downside risk, complementing skewness analysis.
                </Typography>
                <Link href="/education/sortino-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  CAPM Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  Measures systematic risk and complements skewness for a comprehensive risk assessment.
                </Typography>
                <Link href="/education/capm-beta" passHref>
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

export default SkewnessPage;