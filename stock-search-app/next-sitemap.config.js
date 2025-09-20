/** @type {import('next-sitemap').IConfig} */
module.exports = {
  siteUrl:
    process.env.SITE_URL || "https://indportfoliooptimization.vercel.app",
  generateRobotsTxt: true,
  sitemapSize: 5000,
  changefreq: "weekly",
  priority: 0.7,
  exclude: [
    "/api/*",
    "/_next/*",
    // add any private/gated pages here
  ],
  transform: async (config, path) => {
    const high = ["/", "/dividend", "/about", "/docs"];
    return {
      loc: path,
      changefreq: high.includes(path) ? "daily" : config.changefreq,
      priority: high.includes(path) ? 0.9 : config.priority,
      lastmod: new Date().toISOString(),
    };
  },
  robotsTxtOptions: {
    policies:
      process.env.NODE_ENV === "production"
        ? [{ userAgent: "*", allow: "/" }]
        : [{ userAgent: "*", disallow: "/" }],
  },
};
