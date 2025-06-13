import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Divider,
  Chip
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';

const docsIndexPage: React.FC = () => {
  // Define content organized into three distinct categories
  const optimizationMethods = [
    {
      title: 'Mean-Variance Optimization (MVO)',
      description: 'The cornerstone of Modern Portfolio Theory that helps investors construct optimal portfolios balancing risk and return.',
      path: '/docs/mvo',
      available: true
    },
    {
      title: 'Minimum Volatility',
      description: 'A portfolio optimization approach that focuses solely on minimizing risk without a specific return target.',
      path: '/docs/min-vol',
      available: true
    },
    {
      title: 'Maximum Quadratic Utility',
      description: 'Maximizes expected portfolio return minus a penalty for variance, balancing risk aversion with return expectations.',
      path: '/docs/max-quadratic-utility',
      available: true
    },
    {
      title: 'Critical Line Algorithm',
      description: 'An algorithm developed by Harry Markowitz to trace the entire efficient frontier for portfolio optimization problems.',
      path: '/docs/cla',
      available: true
    },
    {
      title: 'Hierarchical Risk Parity (HRP)',
      description: 'A modern portfolio optimization method that uses machine learning techniques to build diversified portfolios.',
      path: '/docs/hrp',
      available: true
    },
    {
      title: 'Hierarchical Equal Risk Contribution (HERC)',
      description: 'An extension of HRP that combines hierarchical clustering with equal risk contribution principles.',
      path: '/docs/herc',
      available: true
    },
    {
      title: 'Nested Clustered Optimization (NCO)',
      description: 'A hybrid approach that combines hierarchical clustering with traditional optimization techniques.',
      path: '/docs/nco',
      available: true
    },
    {
      title: 'Hierarchical Equal Risk Contribution 2 (HERC2)',
      description: 'A simplified hierarchical approach that uses equal weights within clusters for portfolio construction.',
      path: '/docs/herc2',
      available: true
    },
    {
      title: 'Equally Weighted',
      description: 'A simple yet effective diversification approach that assigns equal weight to all assets in a portfolio.',
      path: '/docs/equi-weighted',
      available: true
    },
    {
      title: 'Minimum Conditional Value at Risk (CVaR)',
      description: 'A portfolio optimization method that minimizes the expected loss in the worst-case scenarios beyond the VaR threshold.',
      path: '/docs/min-cvar',
      available: true
    },
    {
      title: 'Minimum Conditional Drawdown at Risk (CDaR)',
      description: 'A portfolio optimization method that minimizes the maximum expected drawdown with a certain confidence level.',
      path: '/docs/min-cdar',
      available: true
    },
    {
      title: 'Technical Indicator Optimization',
      description: 'A signal-based optimization approach using cross-sectional z-scores of technical indicators combined through linear programming.',
      path: '/docs/technical',
      available: true,
      isNew: true
    }
  ];

    const financialConcepts = [    {      title: 'Capital Asset Pricing Model (CAPM)',      description: 'A model that describes the relationship between systematic risk and expected return for assets, particularly stocks.',      path: '/docs/capm',      available: true    },    {      title: 'Modern Portfolio Theory',      description: 'A framework for constructing portfolios that maximize expected return for a given level of market risk.',      path: '/docs/modern-portfolio-theory',      available: true    },    {      title: 'Efficient Frontier',      description: 'The set of optimal portfolios that offer the highest expected return for a defined level of risk.',      path: '/docs/efficient-frontier',      available: true    },    {      title: 'Expected Returns',      description: 'The anticipated return on an investment based on historical data or forward-looking estimates.',      path: '/docs/expected-returns',      available: true    },    {      title: 'Volatility (σ)',      description: 'A statistical measure of the dispersion of returns, usually measured using standard deviation.',      path: '/docs/volatility',      available: true    }
  ];

  const quantitativeMetrics = [
    {
      title: 'Sharpe Ratio',
      description: 'A measure of risk-adjusted return that helps investors understand the return of an investment compared to its risk.',
      path: '/docs/sharpe-ratio',
      available: true
    },
    {
      title: 'CAPM Beta (β)',
      description: 'A measure of systematic risk that represents how an asset moves relative to the overall market.',
      path: '/docs/capm-beta',
      available: true
    },
    {
      title: 'Rolling Beta (β)',
      description: 'A time-series analysis of beta that shows how an asset\'s relationship with the market changes over different periods.',
      path: '/docs/rolling-beta',
      available: true
    },
    {
      title: 'Sortino Ratio',
      description: 'A modification of the Sharpe ratio that only penalizes returns falling below a specified target or required rate of return.',
      path: '/docs/sortino-ratio',
      available: true
    },
    {
      title: 'Skewness',
      description: 'A measure of the asymmetry of the probability distribution of returns about its mean.',
      path: '/docs/skewness',
      available: true
    },
    {
      title: 'Kurtosis',
      description: 'A measure of the "tailedness" of the probability distribution of returns, indicating the presence of extreme values.',
      path: '/docs/kurtosis',
      available: true
    },
    {
      title: 'Jensen\'s Alpha (α)',
      description: 'A risk-adjusted performance measure that represents the average return on a portfolio above or below CAPM predictions.',
      path: '/docs/jensens-alpha',
      available: true
    },
    {
      title: 'Treynor Ratio',
      description: 'A portfolio performance metric that measures returns earned in excess of the risk-free rate per unit of market risk (beta).',
      path: '/docs/treynor-ratio',
      available: true
    },
    {
      title: 'Value at Risk (VaR)',
      description: 'A statistical technique used to measure the level of financial risk within a portfolio over a specific time frame.',
      path: '/docs/value-at-risk',
      available: true
    },
    {
      title: 'Conditional Value at Risk (CVaR)',
      description: 'Also known as Expected Shortfall, measures the expected loss in the worst-case scenarios beyond the VaR threshold.',
      path: '/docs/conditional-value-at-risk',
      available: true
    },
    {
      title: 'Maximum Drawdown',
      description: 'A measure of the largest peak-to-trough decline in a portfolio\'s value, representing the worst-case scenario for an investment.',
      path: '/docs/maximum-drawdown',
      available: true
    },
    {
      title: 'Blume Adjusted Beta',
      description: 'A modified beta calculation that adjusts for the tendency of betas to revert toward the market average over time.',
      path: '/docs/blume-adjusted-beta',
      available: true
    },
    {
      title: 'Entropy',
      description: 'A measure of uncertainty or randomness in portfolio returns, indicating the level of unpredictability in the system.',
      path: '/docs/entropy',
      available: true
    },
    {
      title: 'Omega Ratio (Ω)',
      description: 'A performance measure that evaluates the probability-weighted ratio of gains versus losses for a threshold return.',
      path: '/docs/omega-ratio',
      available: true
    },
    {
      title: 'Calmar Ratio',
      description: 'A performance measurement that uses the ratio of average annual compound rate of return to maximum drawdown.',
      path: '/docs/calmar-ratio',
      available: true
    },
    {
      title: 'Ulcer Index',
      description: 'A volatility measure that captures the depth and duration of drawdowns, focusing on downside movement.',
      path: '/docs/ulcer-index',
      available: true
    },
    {
      title: 'Entropic Value at Risk (EVaR)',
      description: 'A coherent risk measure that provides tighter bounds on tail risk than traditional VaR or CVaR.',
      path: '/docs/evar',
      available: true
    },
    {
      title: 'Gini Mean Difference',
      description: 'A measure of dispersion in returns that evaluates the average absolute difference between all pairs of observations.',
      path: '/docs/gini-mean-difference',
      available: true
    },
    {
      title: 'Drawdown at Risk (DaR)',
      description: 'A risk metric representing the maximum expected drawdown that won\'t be exceeded with a certain confidence level.',
      path: '/docs/dar',
      available: true
    },
    {
      title: 'Conditional Drawdown at Risk (CDaR)',
      description: 'The expected value of drawdown when exceeding the Drawdown at Risk threshold, measuring tail drawdown risk.',
      path: '/docs/cdar',
      available: true
    },
    {
      title: 'Upside Potential Ratio',
      description: 'A performance metric that evaluates upside potential relative to downside risk, focusing on beneficial asymmetry.',
      path: '/docs/upside-potential-ratio',
      available: true
    },
    {
      title: 'Modigliani Risk-Adjusted Performance (M²)',
      description: 'A measure that adjusts portfolio returns to match market volatility, allowing direct comparison with benchmark returns.',
      path: '/docs/modigliani-risk-adjusted',
      available: true
    },
    {
      title: 'Information Ratio',
      description: 'A performance metric that evaluates active return per unit of risk relative to a benchmark index.',
      path: '/docs/information-ratio',
      available: true
    },
    {
      title: 'Sterling Ratio',
      description: 'A risk-adjusted return metric similar to Calmar but using average annual drawdown minus 10% in the denominator.',
      path: '/docs/sterling-ratio',
      available: true
    },
    {
      title: 'V2 Ratio',
      description: 'A relative performance metric that considers both outperformance versus the benchmark and the stability of that outperformance.',
      path: '/docs/v2-ratio',
      available: true
    },
    {
      title: 'Welch Beta',
      description: 'A robust alternative to traditional beta that uses winsorization to reduce the impact of extreme returns.',
      path: '/docs/welch-beta',
      available: true
    },
    {
      title: 'Semi Beta',
      description: 'A downside beta that measures portfolio sensitivity to the benchmark only during down markets.',
      path: '/docs/semi-beta',
      available: true
    },
    {
      title: 'Coskewness',
      description: 'Measures the relationship between portfolio returns and squared market returns, indicating asymmetrical risk.',
      path: '/docs/coskewness',
      available: true
    },
    {
      title: 'Cokurtosis',
      description: 'Measures the relationship between portfolio returns and cubed market returns, indicating tail dependency.',
      path: '/docs/cokurtosis',
      available: true
    },
    {
      title: 'GARCH Beta',
      description: 'A time-varying measure of portfolio beta that accounts for volatility clustering using GARCH models.',
      path: '/docs/garch-beta',
      available: true
    }
  ];

  return (
    <>
      <Head>
        <title>Documentation | QuantPort India - Indian Portfolio Optimization Methods</title>
        <meta name="description" content="Full documentation for QuantPort India. Explore advanced portfolio optimization methods, metrics, and tutorials for Indian equities: NSE, BSE, MVO, CVaR, HERC, NCO, and more." />
        <meta property="og:title" content="Documentation | QuantPort India - Indian Portfolio Optimization Methods" />
        <meta property="og:description" content="Full documentation for QuantPort India. Explore advanced portfolio optimization methods, metrics, and tutorials for Indian equities: NSE, BSE, MVO, CVaR, HERC, NCO, and more." />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs" />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Back to Home Button */}
        <Box sx={{ mb: 4 }}>
          <Link href="/" passHref>
            <Button variant="outlined" color="primary">
              ← Back to Portfolio Optimizer
            </Button>
          </Link>
        </Box>
        
        {/* Title Section */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Portfolio Optimization docs
          </Typography>
          <Typography variant="subtitle1" color="text.secondary" sx={{ maxWidth: '800px', mx: 'auto' }}>
            Explore the theories, methods, and metrics behind modern portfolio optimization
          </Typography>
        </Box>
        
        {/* Introduction */}
        <Paper elevation={2} sx={{ p: 4, mb: 5 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Understanding Portfolio Optimization
          </Typography>
          <Typography paragraph>
            Portfolio optimization is the process of selecting the best portfolio of assets (such as stocks, bonds, and other investments)
            from a set of possible portfolios, according to some objective. Typically, the objective is to maximize expected return for a 
            given level of risk, or to minimize risk for a given level of expected return.
          </Typography>
          <Typography paragraph>
            The docsal resources provided here aim to demystify the complex mathematical and financial concepts behind portfolio
            optimization, making them accessible to both beginners and experienced practitioners. Each article offers intuitive explanations,
            detailed mathematical formulations, practical examples, and academic references.
          </Typography>
        </Paper>
        
        {/* Financial Concepts Section */}
        <Typography variant="h4" component="h2" gutterBottom sx={{ mb: 3, mt: 5 }}>
          Financial Concepts
        </Typography>
        
        <Grid container spacing={3}>
          {financialConcepts.map((topic, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <Card 
                elevation={2}
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: topic.available ? 'translateY(-5px)' : 'none',
                    boxShadow: topic.available ? '0 8px 16px rgba(0,0,0,0.1)' : 'none',
                  }
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h5" component="h3" gutterBottom>
                    {topic.title}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    {topic.description}
                  </Typography>
                </CardContent>
                <CardActions sx={{ p: 2, pt: 0 }}>
                  {topic.available ? (
                    <Link href={topic.path} passHref>
                      <Button variant="contained" color="primary">
                        Learn More
                      </Button>
                    </Link>
                  ) : (
                    <Button variant="outlined" disabled>
                      Coming Soon
                    </Button>
                  )}
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
        
        {/* Optimization Methods Section */}
        <Typography variant="h4" component="h2" gutterBottom sx={{ mb: 3, mt: 5 }}>
          Optimization Methods
        </Typography>
        
        <Grid container spacing={3}>
          {optimizationMethods.map((topic, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <Card 
                elevation={2}
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: topic.available ? 'translateY(-5px)' : 'none',
                    boxShadow: topic.available ? '0 8px 16px rgba(0,0,0,0.1)' : 'none',
                  }
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Typography variant="h5" component="h3">
                      {topic.title}
                    </Typography>
                    {topic.isNew && (
                      <Chip 
                        label="NEW" 
                        color="success" 
                        size="small"
                        sx={{ 
                          fontSize: '0.75rem',
                          fontWeight: 'bold'
                        }} 
                      />
                    )}
                  </Box>
                  <Typography variant="body1" color="text.secondary">
                    {topic.description}
                  </Typography>
                </CardContent>
                <CardActions sx={{ p: 2, pt: 0 }}>
                  {topic.available ? (
                    <Link href={topic.path} passHref>
                      <Button variant="contained" color="primary">
                        Learn More
                      </Button>
                    </Link>
                  ) : (
                    <Button variant="outlined" disabled>
                      Coming Soon
                    </Button>
                  )}
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
        
        {/* Quantitative Metrics Section */}
        <Typography variant="h4" component="h2" gutterBottom sx={{ mb: 3, mt: 5 }}>
          Financial/Quantitative Metrics
        </Typography>
        
        <Grid container spacing={3}>
          {quantitativeMetrics.map((topic, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <Card 
                elevation={2}
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: topic.available ? 'translateY(-5px)' : 'none',
                    boxShadow: topic.available ? '0 8px 16px rgba(0,0,0,0.1)' : 'none',
                  }
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h5" component="h3" gutterBottom>
                    {topic.title}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    {topic.description}
                  </Typography>
                </CardContent>
                <CardActions sx={{ p: 2, pt: 0 }}>
                  {topic.available ? (
                    <Link href={topic.path} passHref>
                      <Button variant="contained" color="primary">
                        Learn More
                      </Button>
                    </Link>
                  ) : (
                    <Button variant="outlined" disabled>
                      Coming Soon
                    </Button>
                  )}
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
        
        {/* Additional Resources Section */}
        <Paper elevation={2} sx={{ p: 4, mt: 5 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Additional Resources
          </Typography>
          <Divider sx={{ my: 2 }} />
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom>
                Recommended Books
              </Typography>
              <ul>
                <li>
                  <Typography paragraph>
                    "Investments" by Bodie, Kane, and Marcus
                  </Typography>
                </li>
                <li>
                  <Typography paragraph>
                    "Modern Portfolio Theory and Investment Analysis" by Elton, Gruber, Brown, and Goetzmann
                  </Typography>
                </li>
                <li>
                  <Typography paragraph>
                    "Portfolio Selection" by Harry Markowitz
                  </Typography>
                </li>
              </ul>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom>
                Academic Journals
              </Typography>
              <ul>
                <li>
                  <Typography paragraph>
                    Journal of Finance
                  </Typography>
                </li>
                <li>
                  <Typography paragraph>
                    Journal of Portfolio Management
                  </Typography>
                </li>
                <li>
                  <Typography paragraph>
                    Financial Analysts Journal
                  </Typography>
                </li>
              </ul>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom>
                Online Courses
              </Typography>
              <ul>
                <li>
                  <Typography paragraph>
                    Coursera: "Investment Management with Python and Machine Learning Specialization"
                  </Typography>
                </li>
                <li>
                  <Typography paragraph>
                    edX: "Portfolio Risk Management"
                  </Typography>
                </li>
                <li>
                  <Typography paragraph>
                    MIT OpenCourseWare: "Analytics of Finance"
                  </Typography>
                </li>
              </ul>
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

export default docsIndexPage; 