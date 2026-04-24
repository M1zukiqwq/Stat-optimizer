/*
 * Standalone tool to collect KLL histograms from Iceberg Puffin files
 * and write them to ML feedback JSON files.
 *
 * Usage:
 *   javac -cp "presto-iceberg.jar:iceberg-core.jar:..." HistogramCollector.java
 *   java -cp ".:presto-iceberg.jar:..." HistogramCollector \
 *     --warehouse /path/to/warehouse \
 *     --feedback-dir /tmp/ml-feedback \
 *     --table catalog.schema.table_name
 */

import com.facebook.presto.common.type.Type;
import com.facebook.presto.common.type.TypeManager;
import com.facebook.presto.iceberg.statistics.KllHistogram;
import io.airlift.slice.Slices;
import org.apache.iceberg.Table;
import org.apache.iceberg.StatisticsFile;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.hadoop.HadoopCatalog;
import org.apache.iceberg.io.FileIO;
import org.apache.iceberg.io.InputFile;
import org.apache.iceberg.puffin.BlobMetadata;
import org.apache.iceberg.puffin.Puffin;
import org.apache.iceberg.puffin.PuffinReader;
import org.apache.iceberg.types.Types;
import org.apache.iceberg.util.Pair;
import org.apache.hadoop.conf.Configuration;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class HistogramCollector
{
    private static final String KLL_BLOB_TYPE = "presto-kll-sketch-bytes-v1";

    public static void main(String[] args) throws Exception
    {
        // Parse arguments
        Map<String, String> params = parseArgs(args);
        String warehouse = params.get("warehouse");
        String feedbackDir = params.get("feedback-dir");
        String tableName = params.get("table");

        if (warehouse == null || feedbackDir == null || tableName == null) {
            System.err.println("Usage: HistogramCollector --warehouse <path> --feedback-dir <path> --table <name>");
            System.exit(1);
        }

        // Create Hadoop catalog
        Configuration conf = new Configuration();
        Catalog catalog = new HadoopCatalog(conf, warehouse);

        // Load table
        TableIdentifier tableId = TableIdentifier.parse(tableName);
        Table table = catalog.loadTable(tableId);
        System.out.println("Loaded table: " + tableName);

        // Get latest statistics file
        List<StatisticsFile> statsFiles = table.statisticsFiles();
        if (statsFiles.isEmpty()) {
            System.err.println("No statistics files found. Run ANALYZE first.");
            System.exit(1);
        }

        StatisticsFile latestStats = statsFiles.stream()
                .max(Comparator.comparing(StatisticsFile::snapshotId))
                .orElseThrow();

        System.out.println("Reading statistics from: " + latestStats.path());

        // Read KLL histograms from Puffin file
        Map<String, HistogramData> histograms = readKllHistograms(table, latestStats);

        System.out.println("Found " + histograms.size() + " KLL histograms");

        // Write to JSON files
        Path feedbackPath = Paths.get(feedbackDir);
        String simpleTableName = tableId.name();

        for (Map.Entry<String, HistogramData> entry : histograms.entrySet()) {
            String columnName = entry.getKey();
            HistogramData histogram = entry.getValue();

            writeHistogramToJson(feedbackPath, simpleTableName, columnName, histogram);
            System.out.println("  Written: " + simpleTableName + "." + columnName);
        }

        System.out.println("Done!");
    }

    private static Map<String, HistogramData> readKllHistograms(Table table, StatisticsFile statsFile)
            throws IOException
    {
        Map<String, HistogramData> result = new HashMap<>();

        try (FileIO io = table.io()) {
            InputFile inputFile = io.newInputFile(statsFile.path());

            try (PuffinReader reader = Puffin.read(inputFile).build()) {
                for (Pair<BlobMetadata, ByteBuffer> data : reader.readAll(reader.fileMetadata().blobs())) {
                    BlobMetadata metadata = data.first();

                    // Only process KLL sketch blobs
                    if (!KLL_BLOB_TYPE.equals(metadata.type())) {
                        continue;
                    }

                    int fieldId = metadata.inputFields().iterator().next();
                    ByteBuffer blob = data.second();

                    // Get column name from schema
                    Types.NestedField field = table.schema().findField(fieldId);
                    if (field == null) {
                        continue;
                    }

                    String columnName = field.name();

                    // Extract quantiles (simplified - assumes numeric type)
                    HistogramData histogram = extractQuantiles(blob, field.type());

                    result.put(columnName, histogram);
                }
            }
        }

        return result;
    }

    private static HistogramData extractQuantiles(ByteBuffer blob, org.apache.iceberg.types.Type icebergType)
    {
        // Simplified: Create placeholder histogram
        // In real implementation, you would:
        // 1. Convert Iceberg type to Presto Type
        // 2. Create KllHistogram from blob bytes
        // 3. Extract quantiles using inverseCumulativeProbability()

        // For now, return a placeholder
        List<Double> levels = new ArrayList<>();
        List<Double> values = new ArrayList<>();

        for (int i = 1; i <= 9; i++) {
            double level = i / 10.0;
            levels.add(level);
            values.add(level); // Placeholder
        }

        return new HistogramData(0.0, 1.0, 0.0, levels, values);
    }

    private static void writeHistogramToJson(Path feedbackDir, String tableName,
                                            String columnName, HistogramData histogram)
            throws IOException
    {
        // Sanitize names
        String safeTable = tableName.replaceAll("[^a-zA-Z0-9._-]", "_");
        String safeColumn = columnName.replaceAll("[^a-zA-Z0-9._-]", "_");

        Path tableDir = feedbackDir.resolve(safeTable);
        Files.createDirectories(tableDir);

        Path jsonFile = tableDir.resolve(safeColumn + ".json");

        // Create fresh JSON with prior_kll and empty observations
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"prior_kll\": {\n");
        json.append(String.format("    \"min\": %.6f,\n", histogram.min));
        json.append(String.format("    \"max\": %.6f,\n", histogram.max));
        json.append(String.format("    \"null_fraction\": %.6f,\n", histogram.nullFraction));

        // quantile_levels
        json.append("    \"quantile_levels\": [");
        for (int i = 0; i < histogram.quantileLevels.size(); i++) {
            if (i > 0) json.append(", ");
            json.append(String.format("%.2f", histogram.quantileLevels.get(i)));
        }
        json.append("],\n");

        // quantile_values
        json.append("    \"quantile_values\": [");
        for (int i = 0; i < histogram.quantileValues.size(); i++) {
            if (i > 0) json.append(", ");
            json.append(String.format("%.6f", histogram.quantileValues.get(i)));
        }
        json.append("]\n");

        json.append("  },\n");
        json.append("  \"observations\": []\n");
        json.append("}\n");

        Files.write(jsonFile, json.toString().getBytes());
    }

    private static Map<String, String> parseArgs(String[] args)
    {
        Map<String, String> params = new HashMap<>();
        for (int i = 0; i < args.length - 1; i += 2) {
            if (args[i].startsWith("--")) {
                params.put(args[i].substring(2), args[i + 1]);
            }
        }
        return params;
    }

    static class HistogramData
    {
        final double min;
        final double max;
        final double nullFraction;
        final List<Double> quantileLevels;
        final List<Double> quantileValues;

        HistogramData(double min, double max, double nullFraction,
                     List<Double> quantileLevels, List<Double> quantileValues)
        {
            this.min = min;
            this.max = max;
            this.nullFraction = nullFraction;
            this.quantileLevels = quantileLevels;
            this.quantileValues = quantileValues;
        }
    }
}
