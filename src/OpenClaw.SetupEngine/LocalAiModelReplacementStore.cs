using System.Collections.Immutable;
using System.Text.Json;
using System.Text.Json.Serialization;
using OpenClaw.Connection.LocalAi;

namespace OpenClaw.SetupEngine;

internal sealed record LocalAiModelReplacementState
{
    public const int CurrentSchemaVersion = 1;

    public int SchemaVersion { get; init; } = CurrentSchemaVersion;
    public required LocalAiInstallManifest PreviousManifest { get; init; }
    public required LocalAiInstallManifest ReplacementManifest { get; init; }
    public required bool RouterPresetExisted { get; init; }
    public byte[]? RouterPreset { get; init; }
    public Uri? PublishedReplacementEndpoint { get; init; }
    public Uri? PendingReplacementEndpoint { get; init; }
}

internal sealed class LocalAiModelReplacementStore
{
    private const int MaximumRouterPresetBytes = 1024 * 1024;
    private static readonly JsonSerializerOptions JsonOptions = CreateJsonOptions();
    private readonly LocalAiPaths _paths;
    private readonly LocalAiManifestStore _manifestStore;

    public LocalAiModelReplacementStore(LocalAiPaths paths)
    {
        _paths = paths ?? throw new ArgumentNullException(nameof(paths));
        _manifestStore = new LocalAiManifestStore(paths);
    }

    public async Task<LocalAiModelReplacementState?> LoadAsync(
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(_paths.ModelReplacementPath))
            return null;

        LocalAiModelReplacementState? state;
        try
        {
            await using var stream = new FileStream(
                _paths.ModelReplacementPath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                16 * 1024,
                FileOptions.Asynchronous | FileOptions.SequentialScan);
            state = await JsonSerializer.DeserializeAsync<LocalAiModelReplacementState>(
                    stream,
                    JsonOptions,
                    cancellationToken)
                .ConfigureAwait(false);
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException(
                "The Local AI model replacement receipt is invalid JSON or uses an unsupported format.",
                ex);
        }

        Validate(state ?? throw new InvalidDataException(
            "The Local AI model replacement receipt is empty."));
        return state;
    }

    public async Task SaveAsync(
        LocalAiModelReplacementState state,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(state);
        Validate(state);
        Directory.CreateDirectory(_paths.RootDirectory);
        _ = _paths.ResolveContainedPath(
            Path.GetFileName(_paths.ModelReplacementPath),
            nameof(_paths.ModelReplacementPath));

        string temporaryPath = Path.Combine(
            _paths.RootDirectory,
            $".{Path.GetFileName(_paths.ModelReplacementPath)}.{Guid.NewGuid():N}.tmp");
        try
        {
            await using (var stream = new FileStream(
                temporaryPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                16 * 1024,
                FileOptions.Asynchronous | FileOptions.WriteThrough))
            {
                await JsonSerializer.SerializeAsync(stream, state, JsonOptions, cancellationToken)
                    .ConfigureAwait(false);
                await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
                stream.Flush(flushToDisk: true);
            }

            _ = _paths.ResolveContainedPath(Path.GetFileName(temporaryPath), nameof(temporaryPath));
            File.Move(temporaryPath, _paths.ModelReplacementPath, overwrite: true);
        }
        finally
        {
            try { File.Delete(temporaryPath); }
            catch { }
        }
    }

    public Task DeleteAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        File.Delete(_paths.ModelReplacementPath);
        return Task.CompletedTask;
    }

    public LocalAiResolvedInstall ResolvePrevious(LocalAiModelReplacementState state)
    {
        Validate(state);
        return _manifestStore.ResolveAndValidate(state.PreviousManifest);
    }

    public LocalAiResolvedInstall ResolveReplacementEndpoint(
        LocalAiModelReplacementState state,
        Uri endpoint)
    {
        Validate(state);
        return _manifestStore.ResolveAndValidate(state.ReplacementManifest with
        {
            Endpoint = endpoint,
        });
    }

    public bool MatchesPrevious(
        LocalAiModelReplacementState state,
        LocalAiInstallManifest manifest) =>
        MatchesImmutableInstall(state.PreviousManifest, manifest);

    public bool MatchesReplacement(
        LocalAiModelReplacementState state,
        LocalAiInstallManifest manifest) =>
        MatchesImmutableInstall(state.ReplacementManifest, manifest);

    private void Validate(LocalAiModelReplacementState state)
    {
        if (state.SchemaVersion != LocalAiModelReplacementState.CurrentSchemaVersion)
        {
            throw new InvalidDataException(
                $"Unsupported Local AI model replacement schema version {state.SchemaVersion}.");
        }

        _ = _manifestStore.ResolveAndValidate(state.PreviousManifest);
        _ = _manifestStore.ResolveAndValidate(state.ReplacementManifest);
        if (string.Equals(
                state.PreviousManifest.ModelCatalogId,
                state.ReplacementManifest.ModelCatalogId,
                StringComparison.Ordinal) ||
            !RuntimeMatches(state.PreviousManifest, state.ReplacementManifest) ||
            !GpuIdsMatch(
                state.PreviousManifest.SelectedGpuId,
                state.ReplacementManifest.SelectedGpuId))
        {
            throw new InvalidDataException(
                "The Local AI model replacement receipt does not describe one runtime and GPU changing models.");
        }

        if (state.RouterPresetExisted != (state.RouterPreset is not null))
        {
            throw new InvalidDataException(
                "The Local AI model replacement router preset snapshot is inconsistent.");
        }
        if (state.RouterPreset is { Length: > MaximumRouterPresetBytes })
        {
            throw new InvalidDataException(
                "The Local AI model replacement router preset snapshot is too large.");
        }

        ValidateReplacementEndpoint(state, state.PublishedReplacementEndpoint);
        ValidateReplacementEndpoint(state, state.PendingReplacementEndpoint);
    }

    private void ValidateReplacementEndpoint(
        LocalAiModelReplacementState state,
        Uri? endpoint)
    {
        if (endpoint is not null)
            _ = _manifestStore.ResolveAndValidate(state.ReplacementManifest with { Endpoint = endpoint });
    }

    private static bool RuntimeMatches(
        LocalAiInstallManifest previous,
        LocalAiInstallManifest replacement) =>
        string.Equals(previous.Engine, replacement.Engine, StringComparison.Ordinal) &&
        string.Equals(previous.EngineVersion, replacement.EngineVersion, StringComparison.Ordinal) &&
        string.Equals(previous.Architecture, replacement.Architecture, StringComparison.Ordinal) &&
        string.Equals(previous.RuntimeId, replacement.RuntimeId, StringComparison.Ordinal) &&
        string.Equals(previous.ExecutablePath, replacement.ExecutablePath, StringComparison.Ordinal) &&
        AssetSetsMatch(previous.RuntimeAssets, replacement.RuntimeAssets);

    private static bool MatchesImmutableInstall(
        LocalAiInstallManifest expected,
        LocalAiInstallManifest actual) =>
        RuntimeMatches(expected, actual) &&
        GpuIdsMatch(expected.SelectedGpuId, actual.SelectedGpuId) &&
        string.Equals(expected.ModelCatalogId, actual.ModelCatalogId, StringComparison.Ordinal) &&
        string.Equals(expected.ModelPath, actual.ModelPath, StringComparison.Ordinal) &&
        string.Equals(expected.ModelId, actual.ModelId, StringComparison.Ordinal) &&
        string.Equals(expected.ModelAlias, actual.ModelAlias, StringComparison.Ordinal) &&
        AssetMatches(expected.ModelAsset, actual.ModelAsset) &&
        expected.RequestedPort == actual.RequestedPort &&
        expected.ContextLength == actual.ContextLength &&
        expected.KeyCachePrecision == actual.KeyCachePrecision &&
        expected.ValueCachePrecision == actual.ValueCachePrecision &&
        expected.DraftKeyCachePrecision == actual.DraftKeyCachePrecision &&
        expected.DraftValueCachePrecision == actual.DraftValueCachePrecision;

    private static bool AssetSetsMatch(
        ImmutableArray<LocalAiAssetReceipt> expected,
        ImmutableArray<LocalAiAssetReceipt> actual) =>
        expected.Length == actual.Length &&
        expected.All(left => actual.Any(right => AssetMatches(left, right)));

    private static bool AssetMatches(LocalAiAssetReceipt left, LocalAiAssetReceipt right) =>
        string.Equals(left.FileName, right.FileName, StringComparison.Ordinal) &&
        string.Equals(left.SourceUrl, right.SourceUrl, StringComparison.Ordinal) &&
        left.SizeBytes == right.SizeBytes &&
        string.Equals(left.Sha256, right.Sha256, StringComparison.Ordinal);

    private static bool GpuIdsMatch(string left, string right) =>
        string.Equals(left, right, StringComparison.Ordinal) ||
        left.StartsWith("cuda:", StringComparison.Ordinal) &&
        string.Equals(left["cuda:".Length..], right, StringComparison.Ordinal) ||
        right.StartsWith("cuda:", StringComparison.Ordinal) &&
        string.Equals(right["cuda:".Length..], left, StringComparison.Ordinal);

    private static JsonSerializerOptions CreateJsonOptions()
    {
        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web)
        {
            WriteIndented = true,
            UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        };
        options.Converters.Add(new JsonStringEnumConverter(
            JsonNamingPolicy.SnakeCaseLower,
            allowIntegerValues: false));
        return options;
    }
}
