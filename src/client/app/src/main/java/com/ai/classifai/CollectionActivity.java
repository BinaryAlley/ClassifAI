package com.ai.classifai;

// #region ================================================================== IMPORTS ====================================================================================
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.recyclerview.widget.RecyclerView;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import com.google.android.material.chip.Chip;
import com.ai.classifai.customview.SquareImageView;
import com.ai.classifai.helpers.Persistence;
import org.tensorflow.lite.examples.classification.R;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
// #endregion

/**
 * Activity to display and manage a collection of images
 * Handles the layout and interaction logic for the image collection
 * 
 * Creation Date: 09th of January, 2021
 */
public class CollectionActivity extends AppCompatActivity {
// #region =============================================================== FIELD MEMBERS =================================================================================
    RecyclerView rv;
// #endregion

// #region ================================================================== METHODS ====================================================================================
    /**
     * Initializes the activity with the collection layout
     * Binds the RecyclerView with the CollectionListAdapter to display image collection
     *
     * @param savedInstanceState If the activity is being re-initialized after being previously shut down, this holds the most recent data provided by onSaveInstanceState(Bundle). Otherwise, it is null.
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_collection);
        rv = findViewById(R.id.rv_collection);
        // setting adapter with the list of collection file paths
        rv.setAdapter(new CollectionListAdapter(Persistence.self().getCollectionFilesPaths()));
    }
// #endregion
}

/**
 * Adapter for the RecyclerView in CollectionActivity
 * 
 * Creation Date: 09th of January, 2021
 */
class CollectionListAdapter extends RecyclerView.Adapter<CollectionItemViewHolder> {
// #region =============================================================== FIELD MEMBERS =================================================================================
    private final ArrayList<String> paths;
// #endregion

// #region ==================================================================== CTOR =====================================================================================
    /**
     * Overload C-tor
     * 
     * @param paths Array of file paths for the images in the collection
     */
    public CollectionListAdapter(String[] paths) {

        this.paths = new ArrayList<String>(Arrays.asList(paths));
    }
// #endregion

// #region ================================================================== METHODS ====================================================================================
    /**
     * Called when RecyclerView needs a new {@link CollectionItemViewHolder} to represent an item
     *
     * @param parent The ViewGroup into which the new View will be added after it is bound to an adapter position
     * @param viewType The view type of the new View
     * @return A new ViewHolder that holds a View of the given view type
     */
    @NonNull
    @Override
    public CollectionItemViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        // inflates layout for each item in the collection       
        return new CollectionItemViewHolder(LayoutInflater.from(parent.getContext())
                .inflate(R.layout.layout_collection_item, parent, false));
    }

    /**
     * Called by RecyclerView to display the data at the specified position
     * This method updates the contents of the {@link CollectionItemViewHolder#itemView} to reflect the item at the given position
     *
     * @param holder The ViewHolder which should be updated to represent the contents of the item at the given position in the data set
     * @param position The position of the item within the adapter's data set
     */
    @Override
    public void onBindViewHolder(@NonNull CollectionItemViewHolder holder, int position) {
        // retrieve the file path for the current position
        String path = paths.get(position);
        // decode the bitmap from the file path
        Bitmap bitmap = BitmapFactory.decodeFile(path);
        // extract file name and title from the path
        String[] filesNames = path.split(File.separator);
        String fileName = filesNames[filesNames.length -1];
        String title = fileName.split("_")[0];
        String label = title.split(" ")[0];
        // capitalize the first letter of the title, if it's not already
        char firstLetter = title.charAt(0);
        if (!Character.isUpperCase(firstLetter))
            title = (firstLetter + "" ).toUpperCase() + title.substring(1);
        String finalTitle = title;
        // set the text and image for the current item    
        holder.caption.setText(title);
        holder.im.setImageBitmap(bitmap);
        // set up click listener for sharing the image
        holder.actionShare.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_SEND);
            intent.setType("image/*");
            intent.putExtra(Intent.EXTRA_STREAM, Uri.fromFile(new File(path)));
            intent.putExtra(Intent.EXTRA_TITLE, finalTitle);
            intent.putExtra(Intent.EXTRA_SUBJECT, finalTitle);
            intent.putExtra(Intent.EXTRA_TEXT, finalTitle);
            holder.actionShare.getContext().startActivity(Intent.createChooser(intent, "Share Image"));
        });
        // set up click listener for deleting the image
        holder.actionDelete.setOnClickListener(v -> {
            if (Persistence.self().delete(path)) {

                notifyItemRemoved(holder.getAdapterPosition());
                paths.remove(path);
            }
        });
        // set up click listener for opening related Wikipedia page
        holder.actionWiki.setOnClickListener(v -> {
            String url = null;
            switch (label.toLowerCase()){
                case "daisy":
                    url = "https://en.wikipedia.org/wiki/Asteraceae";
                    break;
                case "dandelion":
                    url = "https://en.wikipedia.org/wiki/Taraxacum";
                    break;
                case "roses":
                    url = "https://en.wikipedia.org/wiki/Rose";
                    break;
                case "sunflowers":
                    url = "https://en.wikipedia.org/wiki/Helianthus";
                    break;
                case "tulips":
                    url = "https://en.wikipedia.org/wiki/Tulip";
                    break;
            }
            if(url != null){
                Intent i = new Intent(Intent.ACTION_VIEW);
                i.setData(Uri.parse(url));
                holder.actionDelete.getContext().startActivity(i);
            }
        });
    }

    /**
     * Returns the total number of items in the data set held by the adapter
     *
     * @return The total number of items in this adapter
     */    
    @Override
    public int getItemCount() {
        return paths.size();
    }
// #endregion
}

/**
 * ViewHolder for items in the collection list
 * Provides references to the UI components for each item in the collection
 * 
 * Creation Date: 09th of January, 2021
 */
class CollectionItemViewHolder extends RecyclerView.ViewHolder {
// #region =============================================================== FIELD MEMBERS =================================================================================
    SquareImageView im;
    Chip actionShare, actionDelete, actionWiki;
    AppCompatTextView caption;
// #endregion

// #region ==================================================================== CTOR =====================================================================================
    /**
     * Overload C-tor
     * 
     * @param itemView The view of the collection item
     */
    public CollectionItemViewHolder(@NonNull View itemView) {
        super(itemView);
        im = itemView.findViewById(R.id.img_prediction);
        actionShare = itemView.findViewById(R.id.action_share);
        actionDelete = itemView.findViewById(R.id.action_delete);
        actionWiki = itemView.findViewById(R.id.action_wiki);
        caption = itemView.findViewById(R.id.txt_caption);
    }
// #endregion
}