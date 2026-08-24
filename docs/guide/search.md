# Search

**Search** answers: where did matching activity appear?

SurvNG supports ordinary incident filters and, when enabled, **Smart Search** — find incidents from a short visual description.

![Search results for a visual description](images/search-workspace.png)

## Filters

Even without Smart Search you can narrow incidents by camera, object label, zone, and time. That is often enough for “show me cars at the gate yesterday.”

## Smart Search (optional)

Smart Search compares your text to pictures SurvNG already stored. Images and search indexes stay on your SurvNG host. They are not uploaded to the AI assistant provider.

### Example queries

- `person in a red jacket`
- `white delivery truck`
- `dog near the fence`
- `package on the porch`

Concrete visual phrases work best. Abstract ideas (“suspicious,” “threat”) match poorly because the search compares appearance, not intent.

### Setup summary

1. Build or install the Smart Search model package.
2. Enable Smart Search under **Admin → Detection**.
3. Allow SurvNG time to index existing incident pictures.
4. Open **Search**, type a description, and review the ranked results.

Details for model packages: [Smart Search model packages](../semantic-search.md).

## Tips

- Combine a text query with a camera filter when you already know where to look.
- If results look random, the index may still be warming up, or the model package may be missing.
- Smart Search finds object incidents with pictures; it is not a full-text search of filenames.

## Related

- [Incidents](incidents.md)
- [Admin](admin.md)
- [HTTP API](api.md) (`POST /api/semantic-search`)
