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

const KurtosisPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Kurtosis | Portfolio Optimization</title>
        <meta name="description" content="Learn about Kurtosis, a measure of the 'tailedness' of return distributions that helps identify strategies with higher probability of extreme events." />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
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
            Kurtosis
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Understanding the "Tailedness" of Return Distributions
          </Typography>
        </Box>
        
        {/* What Is Kurtosis */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            What Is Kurtosis?
          </Typography>
          <Typography paragraph>
            <strong>Kurtosis</strong> quantifies the <strong>weight of a distribution's tails</strong> relative to the normal bell-curve:
          </Typography>
          
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Name</strong></TableCell>
                  <TableCell><strong>Excess Kurtosis (<InlineMath math="\gamma_2" />)</strong></TableCell>
                  <TableCell><strong>Shape</strong></TableCell>
                  <TableCell><strong>Intuition</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Leptokurtic</strong></TableCell>
                  <TableCell><InlineMath math="\gamma_2 > 0" /></TableCell>
                  <TableCell><em>Peaked</em> centre, <strong>fat tails</strong></TableCell>
                  <TableCell>Higher chance of extreme gains <em>and</em> losses</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Mesokurtic</strong></TableCell>
                  <TableCell><InlineMath math="\gamma_2 = 0" /></TableCell>
                  <TableCell>Normal-like</TableCell>
                  <TableCell>Gaussian benchmark</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Platykurtic</strong></TableCell>
                  <TableCell><InlineMath math="\gamma_2 < 0" /></TableCell>
                  <TableCell>Flatter centre, <strong>thin tails</strong></TableCell>
                  <TableCell>Extremes rarer than Gaussian</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mb: 2, fontStyle: 'italic', borderLeft: '4px solid #2196f3' }}>
            <Typography>
              <strong>Rule of thumb:</strong> A strategy with <InlineMath math="\gamma_2 \gg 0" /> hides more "black-swan" risk than variance alone suggests.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Definition */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Mathematical Definition
          </Typography>
          <Typography paragraph>
            For de-meaned returns <InlineMath math="r_t" /> with standard deviation <InlineMath math="\sigma" />:
          </Typography>
          <Equation math="\text{Excess Kurtosis }(\gamma_2) = \frac{1}{n\sigma^4}\sum_{t=1}^{n} r_t^4 - 3" />
          
          <Box sx={{ pl: 3, mb: 2 }}>
            <ul>
              <li>
                <Typography paragraph>
                  Subtracting <strong>3</strong> converts <em>raw</em> kurtosis to <strong>excess</strong> kurtosis so that a normal distribution sits at 0.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  Fourth power makes the metric hypersensitive to outliers.
                </Typography>
              </li>
            </ul>
          </Box>
        </Paper>
        
        {/* Backend Implementation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Our Backend Implementation
          </Typography>
          <Typography paragraph>
            Inside <strong>srv.py → compute_custom_metrics</strong>:
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
{`kurtosis = port_returns.kurt()`}
          </Box>
          
          <Typography paragraph sx={{ mt: 2 }}>
            <code>pandas.Series.kurt()</code> produces the <strong>bias-adjusted Fisher–Pearson estimator</strong> of excess kurtosis—exactly the <InlineMath math="\gamma_2" /> above.
            Each optimisation method's value is stored in <code>performance.kurtosis</code> and displayed in your results cards.
          </Typography>
        </Paper>
        
        {/* Interpreting Excess Kurtosis */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpreting Excess Kurtosis
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong><InlineMath math="\gamma_2" /></strong></TableCell>
                  <TableCell><strong>Typical Strategy Examples</strong></TableCell>
                  <TableCell><strong>Risk Narrative</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>{'>'} 3</strong> (<em>very fat</em>)</TableCell>
                  <TableCell>Short-vol (option selling), carry trades, stable-coin yields</TableCell>
                  <TableCell>Many small profits punctuated by rare, catastrophic draw-downs</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>≈ 0–1</strong></TableCell>
                  <TableCell>Broad equity indices, balanced funds</TableCell>
                  <TableCell>Tail risk comparable to Gaussian assumption</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>{'<'} 0</strong></TableCell>
                  <TableCell>Certain trend-followers, lottery stocks</TableCell>
                  <TableCell>Extremes are dampened (rare) but centre flattens – performance may be choppy</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          <Typography paragraph>
            Use kurtosis in tandem with <strong>skewness</strong> and <strong>VaR/CVaR</strong> to map full tail risk.
          </Typography>
        </Paper>
        
        {/* Visual Diagnostics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Visual Diagnostics
          </Typography>
          <Typography paragraph>
            Our portfolio optimization app provides histograms to visualize return distributions, which can help identify the kurtosis of your portfolio returns.
          </Typography>
          <Typography paragraph>
            For a more detailed examination, you might consider:
          </Typography>
          <Box sx={{ pl: 3 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>Histogram with log-y axis</strong> clarifies tails.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>QQ-plot</strong> vs. normal line quickly shows tail divergence.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Kurtosis–Time Chart</strong> (rolling 1-year windows) reveals if tail risk is creeping up.
                </Typography>
              </li>
            </ul>
          </Box>
        </Paper>
        
        {/* Why Investors Should Care */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Why Investors Should Care
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Use-Case</strong></TableCell>
                  <TableCell><strong>How Kurtosis Helps</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Stress testing</strong></TableCell>
                  <TableCell>Identify strategies where "10-sigma" events aren't so rare.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Risk budgeting</strong></TableCell>
                  <TableCell>Allocate less capital to highly leptokurtic sleeves unless adequately hedged.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Product disclosure</strong></TableCell>
                  <TableCell>Flag fat-tailed pay-offs to regulators / clients.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Caveats & Best Practice */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Caveats & Best Practice
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Issue</strong></TableCell>
                  <TableCell><strong>Recommendation</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Extreme sample-sensitivity</strong></TableCell>
                  <TableCell>Winsorise or bootstrap to test stability.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Window length trade-off</strong></TableCell>
                  <TableCell>Longer windows → stable estimate; shorter windows → regime detection.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Ambiguous sign</strong></TableCell>
                  <TableCell>Positive kurtosis alone isn't "bad" if accompanied by high right-tail skew—context matters.</TableCell>
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
                  <strong>Taleb, N. N. (2010).</strong> <em>The Black Swan: The Impact of the Highly Improbable.</em> Random House.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Cont, R. (2001).</strong> "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues." <em>Quantitative Finance</em>, 1(2), 223–236.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Scott, D. (1992).</strong> <em>Multivariate Density Estimation.</em> Wiley – Ch. 4 (Higher-moment estimators).
                </Typography>
              </li>
            </ul>
          </Box>
        </Paper>
        
        {/* Conclusion */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography paragraph sx={{ fontStyle: 'italic' }}>
            By exposing kurtosis alongside variance-based and downside-based metrics, our platform equips users to see beyond average volatility, recognising those hidden tail risks where true portfolio disasters—and sometimes spectacular windfalls—originate.
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
                  Skewness
                </Typography>
                <Typography variant="body2" paragraph>
                  Measure of distribution asymmetry that complements kurtosis for understanding return distribution shapes.
                </Typography>
                <Link href="/docs/skewness" passHref>
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
                  Risk-adjusted measure that focuses on downside risk, important for evaluating strategies with high kurtosis.
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
                  Sharpe Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  Standard risk-adjusted return metric that should be complemented with kurtosis analysis for complete risk assessment.
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

export default KurtosisPage; 